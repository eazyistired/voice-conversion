import os
import torch
import numpy as np
import random
import pandas as pd
import torchaudio
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.vocoder.configs import HifiganConfig, WavegradConfig, WavernnConfig
from TTS.vocoder.models.gan import GAN
from TTS.vocoder.models.wavegrad import Wavegrad
from TTS.vocoder.models.wavernn import Wavernn
from pesq import pesq
import soundfile as sf
import pyworld
import pysptk

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 22050   # original dataset and vocoder SR
EVAL_SR = 16000       # required for PESQ
DATASET_DIR = "datasets/custom"
RESULTS_DIR = "results/ablation_study_input_length_wavegrad" # Changed results directory for ablation
VOCODER_MODELS = {
    "hifigan": "models/hifigan/best_model.pth",
    "wavegrad": "models/wavegrad/best_model.pth",
    "wavernn": "models/wavernn/best_model.pth"
}

# --- UTILS ---
def resample(wav, orig_sr, target_sr):
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
    return resampler(wav_tensor).squeeze(0).numpy()

def compute_mcd(ref_wav, gen_wav, sr=SAMPLE_RATE):
    def extract_mcep(wav):
        # Ensure wav is float64 as pyworld expects
        _f0, t = pyworld.harvest(wav.astype(np.float64), sr)
        sp = pyworld.cheaptrick(wav.astype(np.float64), _f0, t, sr)
        mcep = pysptk.sp2mc(sp, order=24, alpha=0.42)
        return mcep

    min_len = min(len(ref_wav), len(gen_wav))
    ref_mcep = extract_mcep(ref_wav[:min_len])
    gen_mcep = extract_mcep(gen_wav[:min_len])
    
    # Ensure both MCEP sequences have at least one frame
    if len(ref_mcep) == 0 or len(gen_mcep) == 0:
        return np.nan # Return NaN if MCD cannot be computed

    min_frames = min(len(ref_mcep), len(gen_mcep))
    diff = ref_mcep[:min_frames] - gen_mcep[:min_frames]
    dist = np.sqrt((diff ** 2).sum(axis=1))
    return (10.0 / np.log(10)) * np.mean(dist)

def load_vocoder(name):
    if name == "hifigan":
        config = HifiganConfig()
        config.load_json("models/hifigan/config.json")
        voc = GAN(config)
    elif name == "wavegrad":
        config = WavegradConfig()
        config.load_json("models/wavegrad/config.json")
        voc = Wavegrad(config)
    elif name == "wavernn":
        config = WavernnConfig()
        config.load_json("models/wavernn/config.json")
        voc = Wavernn(config)
    else:
        raise ValueError(f"Unknown vocoder: {name}")

    voc.load_checkpoint(config, VOCODER_MODELS[name])
    voc.eval().to(DEVICE)
    return voc

# --- SETUP ---
os.makedirs(RESULTS_DIR, exist_ok=True)

print("[INFO] Loading acoustic model (FastPitch)...")
manager = ModelManager()
model_path, config_path, _ = manager.download_model("tts_models/en/ljspeech/fast_pitch")
tts = TTS(model_path=model_path, config_path=config_path).to(DEVICE)

# Define the 3 audio samples for ablation study
# You should choose these files based on their actual length.
# For example, one short, one medium, one long.
# I'm picking arbitrary ones here; replace with your actual choices.
ABLATION_SAMPLES = [
    {"file": "custom_5s.wav",
     "text": "The box was thrown beside the parked truck."}, # Short
    {"file": "custom_15s.wav",
     "text": "I knew Winnie Fred back in the 1990s in Greenwich Village. I first met her at her 19th birthday party which was quite a wild bash. She was a friend of a friend of mine, a guy who was in his 20s."}, # Medium
    {"file": "custom_30s.wav",
     "text": "Winnie Fred was the most vividly alive woman I had ever met in my young life. So one day, looking for inspiration, I asked her, what's the best book you've ever read? She said, oh darling, I could never narrow it down to just one book, because so many books are important to me. But I can tell you my favorite subject. Ten years ago, I began studying the history of ancient Mesopotamia, and it became my passion."} # Long
]

# --- MAIN LOOP FOR ABLATION STUDY ---
results = {}

for vocoder_name in VOCODER_MODELS.keys():
    print(f"\n--- Evaluating {vocoder_name} for Ablation Study ---")
    vocoder = load_vocoder(vocoder_name)
    
    vocoder_results = []

    for sample_info in ABLATION_SAMPLES:
        file_id = sample_info["file"]
        text = sample_info["text"]
        ref_path = os.path.join(DATASET_DIR, f"{file_id}")

        print(f"  Processing {file_id}...")

        try:
            ref_wav, sr = sf.read(ref_path)
            if sr != SAMPLE_RATE:
                raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr} for {file_id}")

            gen_wav = tts.tts(text=text, vocoder=vocoder)
            gen_wav = np.array(gen_wav)

            # MCD
            mcd_val = compute_mcd(ref_wav, gen_wav)

            # PESQ (requires resampling)
            # PESQ requires samples to be between -1 and 1
            ref_resampled = resample(ref_wav, SAMPLE_RATE, EVAL_SR)
            gen_resampled = resample(gen_wav, SAMPLE_RATE, EVAL_SR)

            # Normalize to -1 to 1 for PESQ if not already
            if ref_resampled.max() > 1.0 or ref_resampled.min() < -1.0:
                ref_resampled = ref_resampled / np.max(np.abs(ref_resampled))
            if gen_resampled.max() > 1.0 or gen_resampled.min() < -1.0:
                gen_resampled = gen_resampled / np.max(np.abs(gen_resampled))

            # Ensure both arrays are of sufficient length for PESQ calculation
            min_pesq_len = min(len(ref_resampled), len(gen_resampled))
            
            if min_pesq_len < EVAL_SR * 0.5: # PESQ generally needs at least 0.5 seconds
                pesq_val = np.nan # Not enough data for reliable PESQ
                print(f"    Warning: Skipping PESQ for {file_id} due to short audio length.")
            else:
                # Truncate to minimum length before PESQ calculation
                pesq_val = pesq(EVAL_SR, ref_resampled[:min_pesq_len], gen_resampled[:min_pesq_len], mode="wb")

            vocoder_results.append({
                "file_id": file_id,
                "length_seconds": len(ref_wav) / SAMPLE_RATE, # Add length for comparison
                "mcd": mcd_val,
                "pesq": pesq_val
            })
            print(f"    {file_id} | Length: {len(ref_wav) / SAMPLE_RATE:.2f}s | MCD: {mcd_val:.2f} | PESQ: {pesq_val:.2f}")

        except Exception as e:
            print(f"    Error processing {file_id}: {e}")
            vocoder_results.append({
                "file_id": file_id,
                "length_seconds": np.nan,
                "mcd": np.nan,
                "pesq": np.nan
            })

    results[vocoder_name] = pd.DataFrame(vocoder_results)

# --- COMPARISON AND REPORTING ---
print("\n--- Ablation Study Results ---")
for vocoder_name, df in results.items():
    print(f"\nVocoder: {vocoder_name}")
    print(df.to_string(index=False)) # Print the full DataFrame for each vocoder
    
    # Calculate averages for each vocoder, ignoring NaN values
    avg_mcd = df["mcd"].mean()
    avg_pesq = df["pesq"].mean()
    print(f"  Average MCD: {avg_mcd:.4f}")
    print(f"  Average PESQ: {avg_pesq:.4f}")
    
    # Save results to a text file
    with open(os.path.join(RESULTS_DIR, f"{vocoder_name}_ablation_scores.txt"), "w") as f:
        f.write(f"Ablation Study Results for {vocoder_name}:\n\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\nAverage MCD: {avg_mcd:.4f}\n")
        f.write(f"Average PESQ: {avg_pesq:.4f}\n")

print("\nAblation study complete. Check the 'results/vocoders_ablation' directory for detailed scores.")
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
SAMPLE_RATE = 22050  # original dataset and vocoder SR
EVAL_SR = 16000      # required for PESQ
N_SAMPLES = 50
DATASET_DIR = "datasets/LJSpeech-1.1"
RESULTS_DIR = "vocoders/results"
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
        _f0, t = pyworld.harvest(wav.astype(np.float64), sr)
        sp = pyworld.cheaptrick(wav.astype(np.float64), _f0, t, sr)
        mcep = pysptk.sp2mc(sp, order=24, alpha=0.42)
        return mcep

    min_len = min(len(ref_wav), len(gen_wav))
    ref_mcep = extract_mcep(ref_wav[:min_len])
    gen_mcep = extract_mcep(gen_wav[:min_len])
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

# Load metadata and select N random samples
metadata = pd.read_csv(os.path.join(DATASET_DIR, "metadata.csv"), sep="|", header=None)
metadata.columns = ["file", "text", "_"]
samples = metadata.sample(N_SAMPLES, random_state=42).reset_index(drop=True)

# --- MAIN LOOP ---
for vocoder_name in VOCODER_MODELS.keys():
    print(f"\n--- Evaluating {vocoder_name} ---")
    vocoder = load_vocoder(vocoder_name)
    mcds, pesqs = [], []

    for idx, row in samples.iterrows():
        file_id = row["file"]
        text = row["text"]
        ref_path = os.path.join(DATASET_DIR, "wavs", f"{file_id}.wav")

        ref_wav, sr = sf.read(ref_path)
        if sr != SAMPLE_RATE:
            raise ValueError(f"Expected SR {SAMPLE_RATE}, got {sr}")

        gen_wav = tts.tts(text=text, vocoder=vocoder)
        gen_wav = np.array(gen_wav)

        # MCD
        mcd_val = compute_mcd(ref_wav, gen_wav)

        # PESQ (requires resampling)
        ref_resampled = resample(ref_wav, SAMPLE_RATE, EVAL_SR)
        gen_resampled = resample(gen_wav, SAMPLE_RATE, EVAL_SR)
        pesq_val = pesq(EVAL_SR, ref_resampled, gen_resampled, mode="wb")

        mcds.append(mcd_val)
        pesqs.append(pesq_val)
        print(f"{file_id} | MCD: {mcd_val:.2f} | PESQ: {pesq_val:.2f}")

    avg_mcd = np.mean(mcds)
    avg_pesq = np.mean(pesqs)
    with open(os.path.join(RESULTS_DIR, f"{vocoder_name}_scores.txt"), "w") as f:
        f.write(f"Avg MCD: {avg_mcd:.4f}\n")
        f.write(f"Avg PESQ: {avg_pesq:.4f}\n")

    print(f"[{vocoder_name}] DONE: Avg MCD = {avg_mcd:.2f} | Avg PESQ = {avg_pesq:.2f}")

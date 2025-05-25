import os
import time
import torch
import numpy as np
from scipy.io.wavfile import write
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.vocoder.models.wavernn import Wavernn
from TTS.vocoder.configs import WavernnConfig

# ----------------------------------------
# SETTINGS
# ----------------------------------------
VOCODER_MODEL_PATH = "models/wavernn/best_model.pth"
VOCODER_CONFIG_PATH = "models/wavernn/config.json"
TEMPERATURES = [0.7, 1.0, 1.3]
OUTPUT_DIR = "results/ablation_study_temperature_wavernn"
TEXT = "The quick brown fox jumps over the lazy dog."
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------------------
# Load Acoustic Model
# ----------------------------------------
print("Loading FastPitch TTS model...")
manager = ModelManager()
fp_model_path, fp_config_path, _ = manager.download_model("tts_models/en/ljspeech/fast_pitch")
tts = TTS(model_path=fp_model_path, config_path=fp_config_path).to(DEVICE)

# ----------------------------------------
# Load WaveRNN Vocoder
# ----------------------------------------
print("Loading WaveRNN vocoder...")
voc_config = WavernnConfig()
voc_config.load_json(VOCODER_CONFIG_PATH)
vocoder = Wavernn(voc_config)
vocoder.load_checkpoint(voc_config, VOCODER_MODEL_PATH)
vocoder.to(DEVICE)
vocoder.eval()

# ----------------------------------------
# Inference with Different Temperatures
# ----------------------------------------
for temp in TEMPERATURES:
    print(f"\n--- Inference with temperature={temp} ---")
    start_time = time.time()
    
    # Inject temperature if supported; otherwise modify WaveRNN call
    wav = tts.tts(text=TEXT, vocoder=vocoder, vocoder_args={"temperature": temp})
    elapsed = time.time() - start_time

    # Convert to numpy
    wav = np.array(wav)
    max_val = np.max(np.abs(wav))
    if max_val > 1.0:
        wav = wav / max_val

    # Convert to int16
    wav = (wav * 32767).astype(np.int16)
    filename = os.path.join(OUTPUT_DIR, f"wavernn_temp_{temp}.wav")
    write(filename, tts.synthesizer.output_sample_rate, wav)

    # Real-time factor
    duration = len(wav) / tts.synthesizer.output_sample_rate
    rtf = elapsed / duration
    result_str = (
        f"Audio: {filename}\n"
        f"Duration: {duration:.2f}s | Inference Time: {elapsed:.2f}s | RTF: {rtf:.3f}\n"
    )
    print(result_str)
    # Dump print results to file
    with open(os.path.join(OUTPUT_DIR, "results.txt"), "a") as f:
        f.write(result_str)

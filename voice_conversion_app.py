import torch
import numpy as np
from scipy.io.wavfile import write
from TTS.api import TTS
from TTS.utils.manage import ModelManager
from TTS.vocoder.models.gan import GAN
from TTS.vocoder.models.wavegrad import Wavegrad
from TTS.vocoder.models.wavernn import Wavernn
from TTS.vocoder.configs import HifiganConfig, WavegradConfig, WavernnConfig
import whisper
import torchaudio
import datetime

# -----------------------------
# Settings: select models
# -----------------------------
STT_ENGINE = "whisper"  # Options: "whisper", "wav2vec"
VOCODER_TYPE = "tony_wavegrad"  # Options: "hifigan", "wavegrad", "wavernn", "tony_wavegrad"
INPUT_AUDIO = "datasets/custom/rasism.wav"
TEXT_TO_SPEAK = None  # Leave as None to use STT result
OUTPUT_WAV = "results/output.wav"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# STT Module
# -----------------------------
def transcribe_whisper(audio_path):
    model = whisper.load_model("tiny.en", download_root="models/sst/whisper")
    result = model.transcribe(audio_path)
    return result["text"]

def transcribe_wav2vec(audio_path):
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(DEVICE)
    labels = bundle.get_labels()
    waveform, sr = torchaudio.load(audio_path)
    if sr != bundle.sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, bundle.sample_rate)
    with torch.no_grad():
        emissions, _ = model(waveform.to(DEVICE))
    predicted_ids = torch.argmax(emissions[0], dim=-1)

    # Collapse repeated tokens and remove blank tokens (CTC decoding)
    tokens = []
    prev = -1
    for idx in predicted_ids.cpu().numpy():
        if idx != prev and idx < len(labels):
            tokens.append(labels[idx])
        prev = idx
    transcript = ''.join(tokens).replace("-", "").replace("|", " ").strip()  # '|' is often used as space
    return transcript

# -----------------------------
# Load STT Result
# -----------------------------
if TEXT_TO_SPEAK is None:
    print(f"Transcribing using: {STT_ENGINE}")
    if STT_ENGINE == "whisper":
        TEXT_TO_SPEAK = transcribe_whisper(INPUT_AUDIO)
    elif STT_ENGINE == "wav2vec":
        TEXT_TO_SPEAK = transcribe_wav2vec(INPUT_AUDIO)
    else:
        raise ValueError("Invalid STT_ENGINE")

print(f"Transcribed Text: {TEXT_TO_SPEAK}")

# -----------------------------
# Load TTS + Acoustic Model
# -----------------------------
manager = ModelManager()
fp_model_path, fp_config_path, _ = manager.download_model("tts_models/en/ljspeech/fast_pitch")
tts = TTS(model_path=fp_model_path, config_path=fp_config_path).to(DEVICE)

# -----------------------------
# Load Custom Vocoder
# -----------------------------
def load_vocoder(vocoder_type):
    if vocoder_type == "hifigan":
        config = HifiganConfig()
        config.load_json("models/hifigan/config.json")
        voc = GAN(config)
        voc.load_checkpoint(config, "models/hifigan/best_model.pth")

    elif vocoder_type == "wavegrad":
        config = WavegradConfig()
        config.load_json("models/wavegrad/config.json")
        voc = Wavegrad(config)
        voc.load_checkpoint(config, "models/wavegrad/best_model.pth")

    elif vocoder_type == "wavernn":
        config = WavernnConfig()
        config.load_json("models/wavernn/config.json")
        voc = Wavernn(config)
        voc.load_checkpoint(config, "models/wavernn/best_model.pth")

    elif vocoder_type == "tony_wavegrad":
        config = WavegradConfig()
        config.load_json("models/tony_wavegrad/config.json")
        voc = Wavegrad(config)
        voc.load_checkpoint(config, "models/tony_wavegrad/best_model.pth")

    else:
        raise ValueError("Invalid vocoder type.")
    
    voc.eval()
    voc.to(DEVICE)
    return voc

print(f"Using vocoder: {VOCODER_TYPE}")
vocoder = load_vocoder(VOCODER_TYPE)

# -----------------------------
# Synthesize Speech
# -----------------------------
print(f"Synthesizing: '{TEXT_TO_SPEAK}'")
wav = tts.tts(text=TEXT_TO_SPEAK, vocoder=vocoder)
wav = np.array(wav)

# Normalize if needed
max_val = np.max(np.abs(wav))
if max_val > 1.0:
    print(f"Normalizing waveform (max={max_val:.2f})")
    wav = wav / max_val

wav = (wav * 32767).astype(np.int16)
# Add timestamp, STT model, and vocoder model to output filename for uniqueness
timestamp = datetime.datetime.now().strftime("%d_%H")
output_wav_with_info = OUTPUT_WAV.replace(
    ".wav",
    f"_{timestamp}_stt-{STT_ENGINE}_voc-{VOCODER_TYPE}.wav"
)

write(output_wav_with_info, tts.synthesizer.output_sample_rate, wav)
print(f"Saved output to {output_wav_with_info}")

#!/usr/bin/env python
"""
lj_dataset_tool.py – Build an **LJ‑Speech‑compatible** dataset in two phases
without blocking the microphone for STT.

✔ **Hard‑coded to LJ‑Speech spec**
   * Sample rate: **22 050 Hz**
   * Channels: **mono**
   * Bit depth / subtype: **16‑bit signed PCM (PCM_16)**

The script now ENFORCES these parameters:
• During *record* it captures at the target SR and writes WAVs as `PCM_16`.
• During *transcribe* it *validates every file* and, if necessary, converts
  any stray WAVs to the correct format so your dataset is always train‑ready.

–––– Example workflow –––––––––––––––––––––––––––––––––––––––––––––––––––––––
$ python lj_dataset_tool.py record         \
        --out_dir MyVoice-LJSpeech        \
        --speaker_id 1 --block_seconds 5

$ python lj_dataset_tool.py transcribe     \
        --dataset_dir MyVoice-LJSpeech     \
        --device cuda --model_size base

Dependencies
------------
    pip install sounddevice soundfile librosa openai-whisper  # + faster-whisper for speed
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import List

import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa

# ---------------------------------------------------------------------------
# LJ‑Speech spec constants – ***do NOT change unless you really need to.***
# ---------------------------------------------------------------------------
TARGET_SR: int = 22_050          # Hz
TARGET_CHANNELS: int = 1         # mono
TARGET_SUBTYPE: str = "PCM_16"   # 16‑bit signed integer PCM

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Very light LJ‑style cleaner: collapse spaces, drop punctuation, uppercase."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^A-Za-z0-9' ]+", "", text)
    return text.upper()


def beep(freq: int = 1200, length: float = 0.25) -> None:
    """Simple terminal beep to mark recording start/stop."""
    tone = np.sin(2 * np.pi * np.linspace(0, freq, int(length * TARGET_SR)))
    sd.play(tone, TARGET_SR)
    sd.wait()

# ---------------------------------------------------------------------------
#  Phase 1 – RECORD MODE
# ---------------------------------------------------------------------------

def record_block(duration: float) -> np.ndarray:
    """Record `duration` seconds of mono audio at TARGET_SR (float32 in ±1 range)."""
    print(f"▶ Recording {duration:.1f}s …", end="", flush=True)
    wav = sd.rec(int(duration * TARGET_SR), samplerate=TARGET_SR, channels=TARGET_CHANNELS, dtype="float32")
    sd.wait()
    print(" done.")
    return wav.squeeze()


def record_mode(args: argparse.Namespace):
    out_wav_dir = args.out_dir / "wavs"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    speaker_tag = f"LJ{args.speaker_id:03d}"

    # figure out the next index if resuming
    existing = sorted(out_wav_dir.glob(f"{speaker_tag}-*.wav"))
    next_idx = int(existing[-1].stem.split("-")[1]) + 1 if existing else 1

    print("\nPress Ctrl‑C to stop. Start speaking after the beep…")
    beep()

    try:
        while True:
            wav = record_block(args.block_seconds)
            clip_id = f"{speaker_tag}-{next_idx:04d}"
            wav_path = out_wav_dir / f"{clip_id}.wav"
            sf.write(wav_path, wav, TARGET_SR, subtype=TARGET_SUBTYPE)
            print(f"✓ saved {wav_path.relative_to(args.out_dir)}")
            next_idx += 1
    except KeyboardInterrupt:
        print("\nRecording stopped.")
        sys.exit(0)

# ---------------------------------------------------------------------------
#  Phase 2 – TRANSCRIBE MODE
# ---------------------------------------------------------------------------

def load_whisper(device: str, model_size: str):
    """Import Whisper or faster‑whisper lazily to keep record‑mode light."""
    try:
        from faster_whisper import WhisperModel as _Whisper  # type: ignore
        model = _Whisper(model_size, device=device, compute_type="int8_float16")
        is_fast = True
    except ImportError:
        import whisper as _Whisper  # type: ignore
        model = _Whisper.load_model(model_size, device=device)
        is_fast = False
    print(f"Loaded {'faster-whisper' if is_fast else 'openai-whisper'} [{model_size}] on {device}.")
    return model, is_fast


def ensure_lj_format(wav_path: Path) -> None:
    """Validate a WAV file; if SR/channels/subtype don't match LJ spec, convert in‑place."""
    info = sf.info(str(wav_path))
    if (info.samplerate == TARGET_SR and info.channels == TARGET_CHANNELS and info.subtype == TARGET_SUBTYPE):
        return  # already correct

    print(f"• Converting {wav_path.name} → LJ spec")
    y, sr = librosa.load(str(wav_path), sr=None, mono=False)

    # Handle channels → mono
    if y.ndim > 1:
        y = librosa.to_mono(y)

    # Handle sample‑rate conversion
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

    sf.write(wav_path, y.astype("float32"), TARGET_SR, subtype=TARGET_SUBTYPE)


def transcribe_file(model, wav_path: Path, is_fast: bool) -> str:
    if is_fast:
        segments, _ = model.transcribe(str(wav_path), beam_size=5)
        return " ".join(seg.text for seg in segments).strip()
    result = model.transcribe(str(wav_path), fp16=str(model.device) == "cuda")
    return result["text"].strip()


def transcribe_mode(args: argparse.Namespace):
    wav_dir = args.dataset_dir / "wavs"
    if not wav_dir.exists():
        sys.exit(f"✗ {wav_dir} not found. Did you run record mode?")

    wav_files: List[Path] = sorted(wav_dir.glob("*.wav"))
    if not wav_files:
        sys.exit("✗ No WAV files found to transcribe.")

    # Validate / convert all WAVs first so Whisper sees a uniform format
    for wf in wav_files:
        ensure_lj_format(wf)

    model, is_fast = load_whisper(args.device, args.model_size)

    meta_path = args.dataset_dir / "metadata.csv"
    print(f"Writing transcriptions to {meta_path.relative_to(args.dataset_dir)} (will overwrite).")

    with meta_path.open("w", encoding="utf8", newline="") as meta_f:
        writer = csv.writer(meta_f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for idx, wav_path in enumerate(wav_files, 1):
            clip_id = wav_path.stem  # already LJ###-####
            text_raw = transcribe_file(model, wav_path, is_fast)
            text_norm = normalise(text_raw)
            writer.writerow([clip_id, text_raw, text_norm])
            print(f"[{idx}/{len(wav_files)}] {clip_id}: {text_raw}")
    print("✔ Done – metadata.csv created.")

# ---------------------------------------------------------------------------
#  Entry‑point CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    # record sub‑command
    p_rec = sub.add_parser("record", help="Record microphone audio in fixed‑length blocks; save LJ‑spec WAVs only.")
    p_rec.add_argument("--out_dir", type=Path, required=True, help="Output dataset folder (will be created).")
    p_rec.add_argument("--speaker_id", type=int, default=1, help="Numeric speaker ID → LJxxx prefix.")
    p_rec.add_argument("--block_seconds", type=float, default=5.0, help="Length of each chunk in seconds.")

    # transcribe sub‑command
    p_stt = sub.add_parser("transcribe", help="Run Whisper over saved WAVs and create metadata.csv.")
    p_stt.add_argument("--dataset_dir", type=Path, required=True, help="Folder that contains wavs/ from record phase.")
    p_stt.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for Whisper model.")
    p_stt.add_argument("--model_size", default="base", help="Whisper model size (tiny|base|small|medium|large).")

    args = parser.parse_args()

    if args.cmd == "record":
        record_mode(args)
    elif args.cmd == "transcribe":
        transcribe_mode(args)
    else:
        parser.error("Unknown command: " + str(args.cmd))


if __name__ == "__main__":
    main()

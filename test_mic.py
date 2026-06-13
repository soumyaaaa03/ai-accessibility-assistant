# test_mic.py
# Records 4 seconds of audio using sounddevice
# Saves it as a temporary wav file
# Then transcribes it using faster-whisper
# No PyAudio needed at all

import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
from faster_whisper import WhisperModel

# --- Settings ---
SAMPLE_RATE = 16000   # 16kHz is what Whisper expects
DURATION = 4          # seconds to record

print("⏳ Loading Whisper model (first time takes a minute to download)...")

# We use the 'tiny' model — smallest and fastest, good enough for testing
# int8 means it runs efficiently on CPU
model = WhisperModel("tiny", device="cpu", compute_type="int8")

print("✅ Model loaded")
print("🎤 Speak now — recording for 4 seconds...")

# Record audio from default microphone
# shape will be (SAMPLE_RATE * DURATION, 1) — a 2D array
audio = sd.rec(
    int(SAMPLE_RATE * DURATION),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32"
)

# This line BLOCKS until recording is finished
sd.wait()

print("⏳ Transcribing...")

# Whisper expects a 1D float32 numpy array
# audio is shape (64000, 1) so we flatten it to (64000,)
audio_flat = audio.flatten()

# Transcribe directly from numpy array — no file needed
segments, info = model.transcribe(audio_flat, language="en")

# segments is a generator — we loop through it to get the text
full_text = ""
for segment in segments:
    full_text += segment.text + " "

if full_text.strip():
    print(f"✅ You said: {full_text.strip()}")
else:
    print("❌ Nothing detected — try speaking louder or closer to mic")
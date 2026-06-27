# deaf/transcriber.py
# PURPOSE: Convert live microphone speech into text for deaf users
# INPUT: Continuous microphone audio
# OUTPUT: Stream of transcribed text chunks

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from core.mic_utils import find_best_mic, record_audio

# ── Mic Detection ─────────────────────────────────────────────────
print("🔍 Detecting microphone...")
MIC_DEVICE = find_best_mic()

# ── Model Loading ────────────────────────────────────────────────
print("⏳ Loading Whisper model...")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper model loaded")

SAMPLE_RATE = 16000
CHUNK_DURATION = 4


# ── Record One Chunk ─────────────────────────────────────────────
def record_chunk(duration=CHUNK_DURATION):
    """
    Records audio from the microphone for `duration` seconds.
    Returns a 1D numpy array of audio samples (float32).
    """
    print(f"🎤 Listening for {duration} seconds...")
    return record_audio(duration, MIC_DEVICE)


# ── Transcribe One Chunk ──────────────────────────────────────────
def transcribe_chunk(audio_array):
    """
    Takes a numpy audio array and returns transcribed text.
    vad_filter=True automatically skips silent portions.
    """
    segments, info = model.transcribe(
        audio_array,
        language="en",
        vad_filter=True
    )

    text_parts = []
    for segment in segments:
        text_parts.append(segment.text.strip())

    full_text = " ".join(text_parts).strip()
    return full_text


# ── Live Transcription Loop ───────────────────────────────────────
def live_transcribe(callback, stop_flag):
    """
    Continuously records and transcribes audio until stopped.

    Parameters:
        callback  : function that receives each transcribed text chunk
        stop_flag : function that returns True when we should stop
    """
    while not stop_flag():
        audio = record_chunk()
        text = transcribe_chunk(audio)

        if text:
            callback(text)
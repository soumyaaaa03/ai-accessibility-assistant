# deaf/transcriber.py
# PURPOSE: Convert live microphone speech into text for deaf users
# INPUT: Continuous microphone audio
# OUTPUT: Stream of transcribed text chunks

import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel

# ── Model Loading ────────────────────────────────────────────────
# "small" model = good balance of speed vs accuracy on CPU
# compute_type="int8" makes it faster on CPU (slightly lower precision, fine for speech)
# This loads once when the module is imported — takes a few seconds

print("⏳ Loading Whisper model... (this happens once)")
model = WhisperModel("small", device="cpu", compute_type="int8")
print("✅ Whisper model loaded")

# ── Audio Settings ────────────────────────────────────────────────
SAMPLE_RATE = 16000   # Whisper requires 16kHz audio
CHUNK_DURATION = 4    # seconds — how long each recording chunk is


# ── Record One Chunk ─────────────────────────────────────────────
def record_chunk(duration=CHUNK_DURATION):
    """
    Records audio from the microphone for `duration` seconds.

    Returns:
        A 1D numpy array of audio samples (float32), ready for Whisper.

    How it works:
        sd.rec() starts recording immediately and returns a numpy array
        sd.wait() blocks until recording is finished
        channels=1 = mono audio (Whisper doesn't need stereo)
    """
    print(f"🎤 Listening for {duration} seconds...")

    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32'
    )
    sd.wait()  # wait until recording finishes

    # sd.rec() returns shape (samples, 1) — flatten to 1D for Whisper
    return audio.flatten()


# ── Transcribe One Chunk ──────────────────────────────────────────
def transcribe_chunk(audio_array):
    """
    Takes a numpy audio array and returns transcribed text.

    Parameters:
        audio_array : 1D numpy float32 array of audio samples

    Returns:
        A string of transcribed text (empty string if silence/no speech)

    Notes:
        - model.transcribe() returns a generator of "segments"
        - Each segment has a piece of text with timestamps
        - We just join all segment texts together
        - vad_filter=True automatically skips silent portions —
          this prevents Whisper from "hallucinating" text during silence
    """
    segments, info = model.transcribe(
        audio_array,
        language="en",
        vad_filter=True  # Voice Activity Detection — skips silence
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
        callback  : a function that receives each new transcribed text chunk
                    e.g. callback(text) -> prints it or updates UI
        stop_flag : a function that returns True when we should stop
                    e.g. lambda: should_stop == True
                    (this lets the UI control when to stop the loop)

    This function runs in an infinite loop until stop_flag() returns True.
    Designed to be run inside a background thread.
    """
    while not stop_flag():
        audio = record_chunk()
        text = transcribe_chunk(audio)

        if text:  # only call callback if something was actually said
            callback(text)
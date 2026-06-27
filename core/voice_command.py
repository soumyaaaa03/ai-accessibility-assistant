import sounddevice as sd
import numpy as np
from faster_whisper import WhisperModel
from core.tts import speak

# ---------------- Model ----------------

print("⏳ Loading voice command model...")
_model = WhisperModel(
    "tiny",
    device="cpu",
    compute_type="int8"
)
print("✅ Voice command model ready")

# ---------------- Audio Settings ----------------

SAMPLE_RATE = 16000
COMMAND_DURATION = 5

# Print available microphones
print("\nAvailable Audio Devices:\n")
print(sd.query_devices())

# Use default input device
INPUT_DEVICE = sd.default.device[0]
print(f"\nUsing Input Device: {INPUT_DEVICE}\n")

# ---------------- Commands ----------------

COMMAND_MAP = {
    "detect objects": "DETECT",
    "detect": "DETECT",
    "read text": "OCR",
    "read": "OCR",
    "describe scene": "DESCRIBE",
    "describe": "DESCRIBE",
    "start captions": "CAPTIONS_ON",
    "captions on": "CAPTIONS_ON",
    "stop captions": "CAPTIONS_OFF",
    "captions off": "CAPTIONS_OFF",
    "show gestures": "GESTURES",
    "gestures": "GESTURES",
    "stop": "STOP",
    "help": "HELP",
}

HELP_TEXT = (
    "Available commands are: "
    "detect objects, "
    "read text, "
    "describe scene, "
    "start captions, "
    "stop captions, "
    "show gestures, "
    "stop, "
    "help"
)

# ---------------- Record ----------------

def _record_command():

    print(f"\n🎤 Listening for {COMMAND_DURATION} seconds...")

    audio = sd.rec(
        int(COMMAND_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        device=INPUT_DEVICE,
    )

    sd.wait()

    audio = audio.flatten()

    volume = np.max(np.abs(audio))
    print("Mic Volume:", volume)

    if volume > 0:
        audio = audio / volume

    return audio.astype(np.float32)


# ---------------- Transcribe ----------------

def _transcribe(audio_array):

    segments, info = _model.transcribe(
        audio_array,
        language="en",
        vad_filter=False
    )

    text = " ".join(segment.text for segment in segments)

    return text.lower().strip()


# ---------------- Match ----------------

def match_command(text):

    if not text:
        return None

    for phrase, command in COMMAND_MAP.items():
        if phrase in text:
            print(f"✅ Matched '{phrase}' -> {command}")
            return command

    print("❌ No command matched")
    return None


# ---------------- Listen Once ----------------

def listen_for_command():

    audio = _record_command()

    text = _transcribe(audio)

    print(f"Whisper heard: '{text}'")

    if text == "":
        print("⚠️ No speech detected")
        return None

    return match_command(text)


# ---------------- Continuous ----------------

def continuous_listen(callback, stop_flag):

    speak("Accessibility assistant ready. Say help for available commands.")

    while not stop_flag():

        command = listen_for_command()

        if command == "HELP":
            speak(HELP_TEXT, block=True)

        elif command is not None:
            callback(command)
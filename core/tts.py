# core/tts.py
# PURPOSE: Text-to-speech engine for the entire project
# Every feature that needs to speak uses this module
# WHY pyttsx3: works offline, no API key, runs on CPU, supports Windows

import pyttsx3
import threading

# ── Engine Setup ─────────────────────────────────────────────────
# pyttsx3.init() creates the TTS engine
# On Windows this uses the built-in SAPI5 voice engine
engine = pyttsx3.init()

# Speed of speech — default is ~200, we slow it slightly for clarity
# Useful for blind users who need to catch every word
engine.setProperty("rate", 160)

# Volume — 1.0 is max
engine.setProperty("volume", 1.0)

# ── Voice Selection ──────────────────────────────────────────────
# Windows has multiple voices installed — we pick the first English one
voices = engine.getProperty("voices")
for voice in voices:
    if "english" in voice.name.lower() or "zira" in voice.name.lower() or "david" in voice.name.lower():
        engine.setProperty("voice", voice.id)
        break

# ── Thread Lock ──────────────────────────────────────────────────
# pyttsx3 cannot speak two things at once
# This lock prevents two threads from calling speak() simultaneously
_lock = threading.Lock()

# ── Speak Function ───────────────────────────────────────────────
def speak(text, block=False):
    """
    Speaks the given text aloud.

    Parameters:
        text  : the string to speak
        block : if True, waits until speech finishes before returning
                if False, speaks in background so code keeps running
                
    Use block=True when you need to wait (e.g. reading OCR text)
    Use block=False for real-time detection (don't freeze the camera)
    """
    if not text or not text.strip():
        return

    print(f"🔊 Speaking: {text}")

    if block:
        # Blocking mode — runs in current thread, waits to finish
        with _lock:
            engine.say(text)
            engine.runAndWait()
    else:
        # Non-blocking mode — runs in a background thread
        # This means the camera/detection loop keeps running while speaking
        def _speak():
            with _lock:
                engine.say(text)
                engine.runAndWait()
        t = threading.Thread(target=_speak, daemon=True)
        t.start()


# ── Speak List ───────────────────────────────────────────────────
def speak_list(items, prefix=""):
    """
    Speaks a list of strings one by one.
    Optional prefix is spoken before the list.
    
    Example:
        speak_list(["chair", "person"], prefix="I can see")
        → speaks "I can see chair, person"
    """
    if not items:
        speak("Nothing to report")
        return
    full_text = prefix + " " + ", ".join(items) if prefix else ", ".join(items)
    speak(full_text, block=True)
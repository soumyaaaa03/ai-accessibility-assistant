# tests/test_transcriber.py
# Run this to test live speech-to-text
# Speak into mic — transcribed text appears in terminal every ~4 seconds
# Press Ctrl+C to stop

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from deaf.transcriber import live_transcribe

print("🎤 Live transcription started.")
print("👉 Speak naturally — text appears every ~4 seconds")
print("👉 Press Ctrl+C to stop\n")

# Simple stop mechanism using a mutable list (so it can be changed from outside if needed)
stop = [False]

def on_text(text):
    """Called every time a chunk is transcribed"""
    print(f"📝 {text}")

try:
    live_transcribe(callback=on_text, stop_flag=lambda: stop[0])
except KeyboardInterrupt:
    print("\n✅ Transcription stopped")
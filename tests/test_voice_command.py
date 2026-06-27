# tests/test_voice_command.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.voice_command import listen_for_command
from core.tts import speak

print("🎤 Voice Command Test")
print("Say any of these commands:")
print("  'detect objects' | 'read text' | 'describe scene'")
print("  'start captions' | 'stop captions' | 'show gestures'")
print("  'stop' | 'help'")
print("Press Ctrl+C to quit\n")

speak("Voice command test started. Speak a command.", block=True)

while True:
    try:
        command = listen_for_command()

        if command:
            print(f"✅ Command received: {command}\n")
            speak(f"Command {command} received", block=True)
        else:
            print("⏳ Waiting...\n")

    except KeyboardInterrupt:
        print("\n✅ Test stopped")
        break
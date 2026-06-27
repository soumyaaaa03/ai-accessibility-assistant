# # test_mic_level.py
# import sounddevice as sd
# import numpy as np

# SAMPLE_RATE = 16000
# DURATION = 3

# print("🎤 Recording 3 seconds... SPEAK NOW")
# audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE,
#                channels=1, dtype='float32')
# sd.wait()
# audio = audio.flatten()

# print(f"Max volume level: {np.max(np.abs(audio)):.4f}")
# print(f"Mean volume level: {np.mean(np.abs(audio)):.4f}")

# if np.max(np.abs(audio)) < 0.01:
#     print("❌ Mic is recording silence — check your microphone in Windows settings")
# elif np.max(np.abs(audio)) < 0.05:
#     print("⚠️ Mic level very low — speak louder or increase mic volume in Windows")
# else:
#     print("✅ Mic is picking up audio correctly")


# test_mic_level2.py
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 16000
DURATION = 3

# These are your real mic candidates
candidates = [1, 9, 12]

for device_index in candidates:
    device_name = sd.query_devices(device_index)['name']
    print(f"\n🎤 Testing device [{device_index}]: {device_name}")
    print("   Speak now...")

    try:
        audio = sd.rec(
            int(DURATION * SAMPLE_RATE),
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            device=device_index
        )
        sd.wait()
        audio = audio.flatten()
        max_vol = np.max(np.abs(audio))
        print(f"   Max volume: {max_vol:.4f}", end=" ")

        if max_vol > 0.05:
            print("✅ THIS ONE WORKS — use this index")
        elif max_vol > 0.005:
            print("⚠️ Low but audible — possible candidate")
        else:
            print("❌ Silent")

    except Exception as e:
        print(f"   ❌ Error: {e}")

print("\nDone.")
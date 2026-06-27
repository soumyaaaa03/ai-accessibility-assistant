# tests/test_gesture.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from deaf.gesture import load_gesture_model, recognize_gesture, GestureDebouncer
from core.tts import speak

print("⏳ Loading gesture model...")
recognizer = load_gesture_model("models/gesture_recognizer.task")
debouncer = GestureDebouncer(cooldown=2.0)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("\n✅ Ready. Show hand gestures.")
print("👍 Thumb Up = Yes  |  👎 Thumb Down = No")
print("🖐️ Open Palm = Hello  |  ✌️ Victory = Peace")
print("🤟 ILoveYou = Thank You  |  ✊ Closed Fist = Stop")
print("Press Q to quit\n")

while True:
    try:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            break

        gesture_label, mapped_word, annotated_frame = recognize_gesture(
            frame, recognizer
        )

        if mapped_word and debouncer.should_report(gesture_label):
            print(f"✋ {gesture_label} → 💬 {mapped_word}")
            speak(mapped_word, block=False)

        # Safe display text
        display_text = mapped_word if mapped_word else ""
        if display_text:
            cv2.putText(
                annotated_frame,
                display_text,
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 100),
                3
            )

        cv2.imshow("Gesture Recognition - Q to quit", annotated_frame)

    except Exception as e:
        print(f"❌ Loop error: {type(e).__name__}: {e}")
        # Don't break — keep the camera running even if one frame fails

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("✅ Done")
# tests/test_detector.py
# Opens camera for 10 seconds, detects objects, prints + speaks results

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import time
from blind.detector import detect_objects, format_detections_for_speech, draw_detections
from core.tts import speak

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("📷 Camera opened. Detecting for 10 seconds...")
print("Put objects in front of camera — bottles, chairs, people, phones etc.")

last_spoken = ""
last_speak_time = 0
SPEAK_INTERVAL = 4  # seconds between speech outputs

start = time.time()

while time.time() - start < 10:
    ret, frame = cap.read()
    if not ret:
        break

    detections = detect_objects(frame)
    speech_text = format_detections_for_speech(detections)

    current_time = time.time()
    scene_changed = speech_text != last_spoken
    enough_time_passed = (current_time - last_speak_time) > SPEAK_INTERVAL

    if scene_changed and enough_time_passed and detections:
        speak(speech_text)  # non-blocking, runs in background
        last_spoken = speech_text
        last_speak_time = current_time

    if detections:
        print(f"🔍 {speech_text}")

    annotated = draw_detections(frame, detections)
    cv2.imshow("Object Detection", annotated)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detection test complete")

# Give pyttsx3 background thread time to finish speaking after window closes
time.sleep(2)
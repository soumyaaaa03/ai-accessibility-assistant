# tests/test_ocr.py
# Run this to test OCR
# Opens webcam, press SPACE to capture frame and read text aloud, Q to quit

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from blind.ocr_reader import read_text_from_frame
from core.tts import speak

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("📷 Camera ready.")
print("👉 Point camera at text (book, label, sign)")
print("👉 Press SPACE to read text aloud")
print("👉 Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("OCR Test - Press SPACE to read", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        print("⏳ Reading text... (may take 1-3 seconds)")
        text = read_text_from_frame(frame)

        if text:
            print(f"📄 Extracted text: {text}")
            speak(f"Text says: {text}", block=True)
        else:
            print("❌ No text detected")
            speak("No text found", block=True)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
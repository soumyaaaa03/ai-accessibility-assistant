# run this once in terminal to download the model
import urllib.request
import os

os.makedirs("models", exist_ok=True)
url = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
print("⏳ Downloading MediaPipe gesture model...")
urllib.request.urlretrieve(url, "models/gesture_recognizer.task")
print("✅ Model saved to models/gesture_recognizer.task")
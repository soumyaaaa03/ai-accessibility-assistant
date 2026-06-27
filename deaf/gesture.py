# deaf/gesture.py
import mediapipe as mp
import cv2
import numpy as np
import time

BaseOptions = mp.tasks.BaseOptions
GestureRecognizer = mp.tasks.vision.GestureRecognizer
GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

GESTURE_MAP = {
    "Thumb_Up":    "Yes",
    "Thumb_Down":  "No",
    "Open_Palm":   "Hello",
    "Victory":     "Peace",
    "ILoveYou":    "Thank You",
    "Pointing_Up": "Attention",
    "Closed_Fist": "Stop",
    "None":        ""
}

# 21 landmark connections for drawing hand skeleton
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]


def load_gesture_model(model_path="models/gesture_recognizer.task"):
    options = GestureRecognizerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE
    )
    recognizer = GestureRecognizer.create_from_options(options)
    print("✅ Gesture recognizer loaded")
    return recognizer


def draw_landmarks_on_frame(frame, hand_landmarks_list):
    """
    Draws hand skeleton using pure OpenCV.
    
    hand_landmarks_list comes directly from result.hand_landmarks
    In MediaPipe Task API this is a list of lists — each inner list
    has 21 NormalizedLandmark objects with .x .y .z attributes
    
    We wrap everything in try/except so a drawing failure
    never crashes the main loop
    """
    try:
        h, w = frame.shape[:2]

        for hand_landmarks in hand_landmarks_list:
            # Build pixel coordinate list from normalized landmarks
            points = {}
            for idx, landmark in enumerate(hand_landmarks):
                px = int(landmark.x * w)
                py = int(landmark.y * h)
                # Clamp to frame boundaries — prevents drawing outside frame
                px = max(0, min(w - 1, px))
                py = max(0, min(h - 1, py))
                points[idx] = (px, py)

            # Draw connection lines
            for start_idx, end_idx in HAND_CONNECTIONS:
                if start_idx in points and end_idx in points:
                    cv2.line(frame, points[start_idx], points[end_idx],
                             (0, 200, 255), 2)

            # Draw landmark dots
            for idx, point in points.items():
                cv2.circle(frame, point, 5, (0, 255, 0), -1)

    except Exception as e:
        print(f"⚠️ Drawing error (non-fatal): {e}")

    return frame


def recognize_gesture(frame, recognizer):
    """
    Full pipeline with isolated error handling at every step.
    Returns (gesture_label, mapped_word, annotated_frame)
    Never crashes — always returns something safe.
    """

    # Step 1: Prepare frame safely
    try:
        if frame is None or frame.size == 0:
            return "None", "", frame

        # Ensure correct dtype
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8)

        # Make a copy so we don't modify the original
        display_frame = frame.copy()

        # BGR → RGB conversion
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Force contiguous memory layout — critical for MediaPipe on Windows
        rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    except Exception as e:
        print(f"⚠️ Frame prep error: {e}")
        return "None", "", frame

    # Step 2: Run MediaPipe recognition
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        result = recognizer.recognize(mp_image)

    except Exception as e:
        print(f"⚠️ MediaPipe error: {e}")
        return "None", "", display_frame

    # Step 3: Parse gesture result
    try:
        gesture_label = "None"
        mapped_word = ""

        if result.gestures:
            top_gesture = result.gestures[0][0]
            gesture_label = top_gesture.category_name
            confidence = top_gesture.score

            print(f"   [debug] {gesture_label} | {confidence:.2f}")

            if confidence >= 0.6:
                mapped_word = GESTURE_MAP.get(gesture_label, gesture_label)
            else:
                gesture_label = "None"

    except Exception as e:
        print(f"⚠️ Result parsing error: {e}")
        return "None", "", display_frame

    # Step 4: Draw landmarks
    try:
        if result.hand_landmarks:
            display_frame = draw_landmarks_on_frame(
                display_frame,
                result.hand_landmarks
            )
    except Exception as e:
        print(f"⚠️ Landmark draw error: {e}")
        # Don't return here — we still have the gesture result

    return gesture_label, mapped_word, display_frame


class GestureDebouncer:
    def __init__(self, cooldown=2.0):
        self.last_gesture = ""
        self.last_reported_time = 0
        self.cooldown = cooldown

    def should_report(self, gesture_label):
        now = time.time()
        time_since_last = now - self.last_reported_time

        if gesture_label != self.last_gesture:
            self.last_gesture = gesture_label
            self.last_reported_time = now
            return True

        if time_since_last > self.cooldown:
            self.last_reported_time = now
            return True

        return False
# deaf/collect_gestures.py
# PURPOSE: Collect labeled hand landmark data for gesture recognition
# LIBRARY: cvzone (wraps MediaPipe Hands — same accuracy, simpler code)
# OUTPUT: data/gestures.csv

import cv2
import numpy as np
import csv
import os
from cvzone.HandTrackingModule import HandDetector

# ── Detector Setup ────────────────────────────────────────────────
# HandDetector wraps MediaPipe internally
# maxHands=1 — only track one hand
# detectionCon=0.7 — 70% confidence required to detect a hand

detector = HandDetector(maxHands=1, detectionCon=0.7)

# ── Output File Setup ─────────────────────────────────────────────
DATA_DIR = "data"
CSV_PATH = os.path.join(DATA_DIR, "gestures.csv")
os.makedirs(DATA_DIR, exist_ok=True)

COLUMNS = ["label"] + [f"{axis}{i}" for i in range(21)
                        for axis in ("x", "y", "z")]


# ── Normalization ─────────────────────────────────────────────────
def normalize_landmarks(lm_list):
    """
    lm_list from cvzone looks like:
    [[x0,y0,z0], [x1,y1,z1], ... [x20,y20,z20]]
    21 points, each with x,y,z pixel coordinates

    We normalize so gesture works at any hand size or distance:
    1. Make wrist the origin (subtract point 0 from all points)
    2. Scale by wrist-to-middle-fingertip distance (point 12)
    """
    points = np.array(lm_list, dtype=float)

    # Keep only x,y,z — cvzone sometimes gives 4 values, we need 3
    points = points[:, :3]

    # Step 1: wrist becomes origin
    wrist = points[0].copy()
    points -= wrist

    # Step 2: scale normalization
    scale = np.linalg.norm(points[12])
    if scale < 1e-6:
        scale = 1e-6
    points /= scale

    return points.flatten().tolist()


# ── Main Collection Loop ──────────────────────────────────────────
def collect():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Camera not found")
        return

    file_exists = os.path.exists(CSV_PATH)
    csv_file = open(CSV_PATH, "a", newline="")
    writer = csv.writer(csv_file)
    if not file_exists:
        writer.writerow(COLUMNS)

    label_map = {
        ord('1'): "hello",
        ord('2'): "yes",
        ord('3'): "no",
        ord('4'): "thankyou"
    }

    current_label = None
    counts = {"hello": 0, "yes": 0, "no": 0, "thankyou": 0}

    print("=" * 50)
    print("GESTURE DATA COLLECTION")
    print("=" * 50)
    print("1 → hello  |  2 → yes  |  3 → no  |  4 → thankyou")
    print("0 → pause  |  Q → quit and save")
    print("Collect 100+ samples per gesture")
    print("=" * 50)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip for mirror view — more natural to use
        frame = cv2.flip(frame, 1)

        # findHands does detection + draws skeleton on frame automatically
        # returns list of hand dicts, and the annotated frame
        detected_hands, frame = detector.findHands(frame)

        if detected_hands and current_label is not None:
            # lmList = list of 21 landmarks, each [x, y, z]
            lm_list = detected_hands[0]["lmList"]
            features = normalize_landmarks(lm_list)
            writer.writerow([current_label] + features)
            counts[current_label] += 1

        # ── Status display ────────────────────────────────────────
        status = f"Collecting: {current_label}" if current_label else "Paused — press 1-4"
        color = (0, 255, 0) if current_label else (0, 0, 255)
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        y = 60
        for gesture, count in counts.items():
            cv2.putText(frame, f"{gesture}: {count}", (10, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 1)
            y += 25

        cv2.imshow("Gesture Collection", frame)
        key = cv2.waitKey(1) & 0xFF

        if key in label_map:
            current_label = label_map[key]
            print(f"▶ Collecting: {current_label}")
        elif key == ord('0'):
            current_label = None
            print("⏸ Paused")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    csv_file.close()
    print("\n✅ Saved to data/gestures.csv")
    print("Final counts:", counts)


if __name__ == "__main__":
    collect()
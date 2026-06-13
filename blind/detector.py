# blind/detector.py
# PURPOSE: Detect objects in a webcam frame using YOLOv8
# INPUT: A single image frame (numpy array from OpenCV)
# OUTPUT: List of detected objects with label, confidence, position

from ultralytics import YOLO
import cv2

# ── Model Loading ───────────────────────────────────────────────
# YOLOv8n = nano variant, smallest and fastest, good for CPU
# On first run this downloads ~6MB weights file automatically
# After that it loads from cache — no internet needed

model = YOLO("yolov8n.pt")

# ── Position Helper ─────────────────────────────────────────────
def get_position(x_center, frame_width):
    """
    Figures out if the object is on the left, center, or right
    of the frame based on where its center point falls.
    
    We divide the frame into 3 equal zones:
    |-- left --|-- center --|-- right --|
    """
    one_third = frame_width / 3
    two_thirds = (frame_width / 3) * 2

    if x_center < one_third:
        return "left"
    elif x_center > two_thirds:
        return "right"
    else:
        return "center"


# ── Main Detection Function ──────────────────────────────────────
def detect_objects(frame):
    """
    Takes a single webcam frame and returns a list of detected objects.
    
    Each detected object in the returned list looks like:
    {
        "label": "person",
        "confidence": 0.91,
        "position": "left",
        "box": [x1, y1, x2, y2]
    }
    
    Only returns detections with confidence > 0.5 to filter weak guesses.
    """

    # frame is a numpy array (H x W x 3) in BGR format from OpenCV
    # YOLO accepts this directly — no preprocessing needed

    results = model(frame, verbose=False)
    # verbose=False silences the per-frame console output from YOLO

    detections = []

    # results is a list — one element per image passed
    # Since we pass one frame at a time, we only care about results[0]
    for result in results[0].boxes:

        # confidence is how sure YOLO is about this detection (0.0 to 1.0)
        confidence = float(result.conf[0])

        # Skip weak detections — below 50% confidence is unreliable
        if confidence < 0.5:
            continue

        # cls is the class index — an integer like 0, 1, 2...
        # model.names maps that integer to a human label like "person"
        class_id = int(result.cls[0])
        label = model.names[class_id]

        # xyxy gives us the bounding box corners:
        # x1,y1 = top-left corner, x2,y2 = bottom-right corner
        box = result.xyxy[0].tolist()
        x1, y1, x2, y2 = box

        # Calculate center x of the bounding box
        x_center = (x1 + x2) / 2

        # Get frame width from the frame shape
        # frame.shape returns (height, width, channels)
        frame_width = frame.shape[1]

        position = get_position(x_center, frame_width)

        detections.append({
            "label": label,
            "confidence": round(confidence, 2),
            "position": position,
            "box": [int(x1), int(y1), int(x2), int(y2)]
        })

    return detections


# ── Format for Speech ────────────────────────────────────────────
def format_detections_for_speech(detections):
    """
    Converts the detections list into a natural spoken sentence.
    
    Example output:
    "Detected: person on the left, chair in the center, bottle on the right"
    
    If nothing detected, returns a default message.
    """
    if not detections:
        return "No objects detected"

    # Remove duplicate labels — if 2 chairs detected, say "chair" once
    seen = set()
    unique = []
    for d in detections:
        if d["label"] not in seen:
            seen.add(d["label"])
            unique.append(d)

    parts = [f"{d['label']} on the {d['position']}" for d in unique]
    return "Detected: " + ", ".join(parts)


# ── Draw Boxes on Frame ──────────────────────────────────────────
def draw_detections(frame, detections):
    """
    Draws bounding boxes and labels on the frame for visual display.
    Returns the annotated frame.
    This is only for the UI later — not needed for backend logic.
    """
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = f"{d['label']} ({d['confidence']})"

        # Draw rectangle — (frame, top-left, bottom-right, color BGR, thickness)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label text above the box
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame
# blind/ocr_reader.py
# PURPOSE: Extract text from a webcam frame using OCR
# INPUT: A single image frame (numpy array from OpenCV)
# OUTPUT: Combined readable text string

import easyocr

# ── Reader Initialization ────────────────────────────────────────
# This loads the OCR model into memory — takes a few seconds on first call
# gpu=False forces CPU mode (we have no GPU)
# 'en' = English language model
# IMPORTANT: We create this ONCE at module load, not every time we call OCR
# Loading it repeatedly would be very slow

print("⏳ Loading OCR model... (this happens once)")
reader = easyocr.Reader(['en'], gpu=False)
print("✅ OCR model loaded")


# ── Main OCR Function ─────────────────────────────────────────────
def read_text_from_frame(frame, min_confidence=0.4):
    """
    Takes a webcam frame and returns extracted text.

    Parameters:
        frame          : numpy array (image) from OpenCV
        min_confidence : ignore text detections below this confidence
                          (EasyOCR sometimes reads noise as random characters
                           with very low confidence — we filter these out)

    Returns:
        A single string combining all detected text, in reading order
        (top to bottom, left to right — EasyOCR returns it roughly in this order)
    """

    # readtext() does everything: detects text regions AND reads characters
    results = reader.readtext(frame)

    if not results:
        return ""

    extracted_lines = []

    for (bbox, text, confidence) in results:
        # Skip low-confidence garbage detections
        if confidence < min_confidence:
            continue

        # Skip empty or whitespace-only results
        if not text.strip():
            continue

        extracted_lines.append(text.strip())

    # Join all detected text lines with spaces to form readable sentence
    full_text = " ".join(extracted_lines)
    return full_text


# ── Draw Boxes on Frame (for UI later) ───────────────────────────
def draw_ocr_boxes(frame, min_confidence=0.4):
    """
    Draws boxes around detected text regions for visual feedback.
    Returns the annotated frame.
    Used only for UI display — not needed for backend logic.
    """
    import cv2

    results = reader.readtext(frame)

    for (bbox, text, confidence) in results:
        if confidence < min_confidence:
            continue

        # bbox is 4 corner points — convert to integer coordinates
        points = [(int(p[0]), int(p[1])) for p in bbox]

        # Draw a polygon around the detected text
        cv2.polylines(frame, [cv2.UMat(np.array(points))] if False else
                      [__import__('numpy').array(points)],
                      isClosed=True, color=(255, 0, 0), thickness=2)

        # Put the recognized text above the box
        cv2.putText(frame, text, points[0],
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return frame
# test_camera.py
# This opens your webcam and shows the live feed in a window
# Press Q to quit
# Purpose: confirm OpenCV can access your camera before we build on top of it

import cv2

# 0 means the default/first camera on your system
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Camera not found. Check if webcam is connected.")
else:
    print("✅ Camera opened successfully. Press Q to quit.")

while True:
    # Read one frame from camera
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Failed to grab frame")
        break
    
    # Show the frame in a window called "Camera Test"
    cv2.imshow("Camera Test", frame)
    
    # Wait 1ms for a key press — if Q is pressed, exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Always release camera and close windows when done
cap.release()
cv2.destroyAllWindows()
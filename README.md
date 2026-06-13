# AI-Powered Accessibility Assistant

A desktop prototype assistant for blind and deaf users, built with Python and pre-trained AI models.

## Features (In Progress)
- Object detection with voice feedback (YOLOv8 + pyttsx3)
- OCR text reading aloud (EasyOCR)
- Live speech-to-text subtitles (faster-whisper)
- Hand gesture recognition (MediaPipe) — in progress

## Tech Stack
- Python 3.11
- OpenCV, Ultralytics YOLOv8
- EasyOCR
- faster-whisper, sounddevice
- pyttsx3
- Streamlit (UI)

## Setup
\`\`\`bash
python -m venv venv
venv\\Scripts\\activate
pip install -r requirements.txt
\`\`\`

## Status
🚧 Under active development — backend modules being built and tested individually before UI integration.
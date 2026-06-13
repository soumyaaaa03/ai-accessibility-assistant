# verify.py
# This script checks every library we need is correctly installed
# Run it once after installation — if no errors, you're good to go

import cv2
print("✅ OpenCV:", cv2.__version__)

import easyocr
print("✅ EasyOCR imported")

import pyttsx3
engine = pyttsx3.init()
print("✅ pyttsx3 working")

import speech_recognition as sr
print("✅ SpeechRecognition imported")

import mediapipe as mp
print("✅ MediaPipe imported")

import streamlit as st
print("✅ Streamlit imported")

import numpy as np
print("✅ NumPy:", np.__version__)

from ultralytics import YOLO
print("✅ Ultralytics YOLO imported")

from faster_whisper import WhisperModel
print("✅ faster-whisper imported")

import sklearn
print("✅ scikit-learn:", sklearn.__version__)

print("\n✅ All libraries verified. You are ready to build.")
import joblib
from extract_features import extract_mfcc
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

EMOTIONS = ["happy", "sad", "angry", "neutral"]
model = joblib.load("emotion_model.pkl")

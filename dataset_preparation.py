import os
import numpy as np
from extract_features import extract_mfcc

DATASET_PATH = "data"  # folder with happy/sad/angry/neutral
EMOTIONS = ["happy", "sad", "angry", "neutral"]

X = []
y = []

for emotion in EMOTIONS:
    emotion_path = os.path.join(DATASET_PATH, emotion)
    label = EMOTIONS.index(emotion)

    for file in os.listdir(emotion_path):
        if file.endswith(".wav"):
            file_path = os.path.join(emotion_path, file)
            features = extract_mfcc(file_path)
            if features is not None:
                X.append(features)
                y.append(label)

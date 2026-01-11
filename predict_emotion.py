import joblib
from extract_features import extract_mfcc
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np

EMOTIONS = ["happy", "sad", "angry", "neutral"]
model = joblib.load("emotion_model.pkl")
duration = 10  # seconds
fs = 22050    # sample rate

print(f"Recording {duration} seconds of audio. Please speak now...")
audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
sd.wait()
audio = np.squeeze(audio)  # remove single-dimensional entries
temp_file = "temp_audio.wav"
write(temp_file, fs, audio)
features = extract_mfcc(temp_file)



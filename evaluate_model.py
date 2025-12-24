import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
model = joblib.load("emotion_model.pkl")

y_pred = model.predict(X)

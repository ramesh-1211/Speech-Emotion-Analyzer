import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
model = joblib.load("emotion_model.pkl")
y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
print("\nClassification Report:\n")
print(classification_report(y, y_pred, target_names=["happy", "sad", "angry", "neutral"]))
print("Confusion Matrix:\n")
print(confusion_matrix(y, y_pred))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
X = np.load("X_features.npy")
y = np.load("y_labels.npy")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Pipeline([("scaler", StandardScaler()),("svm", SVC(kernel="rbf", probability=True))])
model.fit(X_train, y_train)

joblib.dump(model, "emotion_model.pkl")
print("Model trained and saved successfully!")

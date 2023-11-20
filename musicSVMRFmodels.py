import os
import pandas as pd
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Define paths (adjust as necessary)
audio_base_path = 'D:/MusicGenreClassification/ismir04_genre/audio/training/'
metadata_csv_path = 'D:/MusicGenreClassification/ismir04_genre/metadata/training/tracklistwav.csv'

# Read metadata
metadata = pd.read_csv(metadata_csv_path)

# Initialize lists for features and labels
features = []
labels = []

# Feature extraction
for index, row in metadata.iterrows():
    file_path = os.path.join(audio_base_path, row['file_path'].replace('\\', '/'))
    
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    if pd.isnull(row['class']):
        print(f"Genre not found for file: {file_path}")
        continue

    try:
        y, sr = librosa.load(file_path, mono=True, duration=120)
        feature = librosa.feature.chroma_stft(y=y, sr=sr)
        feature = np.mean(feature, axis=1)
        features.append(feature)
        labels.append(row['class'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# SVM Classifier
print("Training SVM Classifier...")
svm_model = SVC(kernel='linear', C=1)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_val)
print("SVM Classifier Report:")
print(classification_report(y_val, y_pred_svm, target_names=le.classes_))

# Random Forest Classifier
print("Training Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
print("Random Forest Classifier Report:")
print(classification_report(y_val, y_pred_rf, target_names=le.classes_))

import numpy as np
import pandas as pd
import os
import librosa
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import expon, randint

# Define paths
audio_base_path = 'D:/MusicGenreClassification/ismir04_genre/audio/training/'
metadata_csv_path = 'D:/MusicGenreClassification/ismir04_genre/metadata/training/tracklistwav.csv'

# Read metadata
metadata = pd.read_csv(metadata_csv_path)

# Initialize lists for features and labels
features = []
labels = []

# Define the length of your feature vector
feature_length = 20

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
        y, sr = librosa.load(file_path, mono=True, duration=60)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        feature = np.mean(chroma_stft, axis=1)

        # Ensure all feature vectors have the same length
        if len(feature) > feature_length:
            feature = feature[:feature_length]
        elif len(feature) < feature_length:
            feature = np.pad(feature, (0, feature_length - len(feature)))

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

# Split dataset
X_train, X_val, y_train, y_val = train_test_split(features, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded)

# SVM Classifier
print("Performing Randomized Search for SVM...")
param_dist_svm = {'C': expon(scale=100), 'gamma': expon(scale=.1),
                  'kernel': ['rbf', 'linear','sigmoid','poly'],
                  'gamma':['scale', 'auto']}
svm_model = SVC()
random_search_svm = RandomizedSearchCV(svm_model, param_distributions=param_dist_svm, n_iter=20, cv=5, verbose=1, n_jobs=-1)
random_search_svm.fit(X_train, y_train)
print("Best parameters found for SVM: ", random_search_svm.best_params_)
y_pred_svm = random_search_svm.predict(X_val)
print("SVM Classifier Report:")
print(classification_report(y_val, y_pred_svm, target_names=le.classes_))

# Random Forest Classifier
print("Performing Randomized Search for Random Forest...")
param_dist_rf = {"max_depth": [6, None],
              "max_features": randint(1, 11),
              "min_samples_split": randint(2, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}
rf_model = RandomForestClassifier()
random_search_rf = RandomizedSearchCV(rf_model, param_distributions=param_dist_rf, n_iter=20, cv=5, verbose=1, n_jobs=-1)
random_search_rf.fit(X_train, y_train)
print("Best parameters found for Random Forest: ", random_search_rf.best_params_)
y_pred_rf = random_search_rf.predict(X_val)
print("Random Forest Classifier Report:")
print(classification_report(y_val, y_pred_rf, target_names=le.classes_))

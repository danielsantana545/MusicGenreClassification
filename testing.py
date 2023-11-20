import os
import pandas as pd
import numpy as np
import librosa
import time
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.callbacks import ModelCheckpoint
from keras import optimizers

print("Starting program...")

# Define paths
audio_base_path = 'D:/MusicGenreClassification/ismir04_genre/audio/training/'
metadata_csv_path = 'D:/MusicGenreClassification/ismir04_genre/metadata/training/tracklistwav.csv'
model_save_path = 'D:/MusicGenreClassification/models/music_genre_model.h5'

# Read metadata
print("Reading metadata...")
start_time = time.time()
metadata = pd.read_csv(metadata_csv_path)
print(f"Metadata read in {time.time() - start_time:.2f} seconds.")

# Initialize lists for features and labels
features = []
labels = []

# Feature extraction
print("Starting feature extraction...")
start_time = time.time()
for index, row in metadata.iterrows():
    file_path = os.path.join(audio_base_path, row['file_path'].replace('\\', '/'))
    
    if index % 100 == 0 and index > 0:
        print(f"Processed {index} files. Time elapsed: {time.time() - start_time:.2f} seconds.")

    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        continue
    
    if pd.isnull(row['class']):
        print(f"Genre not found for file: {file_path}")
        continue

    try:
        y, sr = librosa.load(file_path, mono=True, duration=120)
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr))
        rmse = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
        feature = np.hstack((chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc))
        features.append(feature)
        labels.append(row['class'])
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        continue

print(f"Feature extraction completed. Total time: {time.time() - start_time:.2f} seconds.")
print(f"Total files processed: {len(features)}")

if not features:
    raise ValueError("No features extracted. Check the audio file paths and feature extraction process.")

# Convert lists to numpy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels
le = LabelEncoder()
yy = to_categorical(le.fit_transform(labels))

# Reshape features for LSTM layer
features = np.reshape(features, (features.shape[0], 1, features.shape[1]))

# Split dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(features, yy, test_size=0.3, random_state=42, stratify=yy)

# Build model
print("Building and training model...")
model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(64))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(len(le.classes_), activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Checkpoint to save best model
checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)

# Train model
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, y_val), callbacks=[checkpoint])

# Load the best saved model
model.load_weights(model_save_path)

# Evaluate the model on validation set
loss, accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

# Predictions
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

# Classification Report
report = classification_report(y_true_classes, y_pred_classes, target_names=le.classes_)
print(report)


""" def build_model(hp):
    model = Sequential()
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   input_shape=(X_train.shape[1], X_train.shape[2]),
                   return_sequences=True,
                   activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'], default='tanh')))
    model.add(Dropout(hp.Float('dropout_1', min_value=0, max_value=0.5, default=0.25, step=0.05)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), 
                   return_sequences=True,
                   activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'], default='tanh')))
    model.add(Dropout(hp.Float('dropout_2', min_value=0, max_value=0.5, default=0.25, step=0.05)))
    model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32),
                   activation=hp.Choice('activation', values=['relu', 'tanh', 'sigmoid'], default='tanh')))
    model.add(Dropout(hp.Float('dropout_3', min_value=0, max_value=0.5, default=0.25, step=0.05)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(hp.Float('dropout_4', min_value=0, max_value=0.5, default=0.25, step=0.05)))
    model.add(Dense(len(le.classes_), activation='softmax'))
    
    model.compile(optimizer=optimizers.Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='my_dir',
    project_name='hparam_tuning'
)

tuner.search_space_summary()

tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
#The hyperparameter search is complete.
#The optimal number of units in the LSTM layers is {best_hps.get('units')}.
#The optimal activation function for the LSTM layers is {best_hps.get('activation')}.
#The optimal dropout rates are {best_hps.get('dropout_1')}, {best_hps.get('dropout_2')}, {best_hps.get('dropout_3')}, and {best_hps.get('dropout_4')}.
#The optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")  """
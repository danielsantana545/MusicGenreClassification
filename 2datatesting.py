import os
import pandas as pd
import numpy as np
import librosa
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import load_img, img_to_array



def extract_features(file_path, label):
    """
    Extracts various features from an audio file.
    """
    try:
        # Get the duration of the audio file
        duration = librosa.get_duration(path=file_path)
        
        # Adjust the offset and duration if the file is too short
        max_duration = 30  # seconds
        if duration > max_duration + 30:
            offset = 30
            actual_duration = max_duration
        else:
            offset = 0
            actual_duration = duration
        
        y, sr = librosa.load(file_path, mono=True, duration=actual_duration, offset=offset)
        
        if not y.any():  # If the loaded audio is silent or empty
            print(f"File {file_path} is silent or too short. Skipping.")
            return None, None

        # Feature extraction
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
        rmse = np.mean(librosa.feature.rms(y=y))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y))
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        spec_contrast = np.mean(librosa.feature.spectral_contrast(S=np.abs(librosa.stft(y)), sr=sr), axis=1)
        harmony, percussiveness = librosa.effects.hpss(y)

        # Concatenate all features into one array
        features = np.hstack((chroma_stft, rmse, spec_cent, spec_bw, rolloff, zcr, mfcc, tempo, spec_contrast, np.mean(harmony), np.mean(percussiveness)))

        return features, label

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None, None
    
def preprocess_data(audio_base_path, metadata_csv_path):
    """
    Preprocesses the audio data.
    """
    start_time = time.time()
    print("Preprocessing data...")

    metadata = pd.read_csv(metadata_csv_path)
    features = []
    labels = []
    print("Staring Feature Extraction...")
    start_time = time.time()
    for index, row in metadata.iterrows():
        file_path = os.path.join(audio_base_path, row['file_path'].replace('\\', '/'))
        if index % 100 ==0 and index > 0:
            print(f"Processed {index} files. Time elapsed: {time.time()-start_time:.2f} seconds.")

        if not os.path.exists(file_path) or pd.isnull(row['class']):
            continue
        feature, label = extract_features(file_path, row['class'])
        if feature is not None:
            features.append(feature)
            labels.append(label)
    
    print("Preprocessing completed in {:.2f} seconds".format(time.time() - start_time))
    return np.array(features), np.array(labels)

# Feature importance analysis function
def analyze_feature_importance(features, labels):
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(features, labels)
    importances = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
    
    feature_names = [f'Feature {i+1}' for i in range(features.shape[1])]
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances,
        'Std': std
    })
    
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, xerr=std)
    plt.title('Feature Importance using RandomForestClassifier')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df

def build_and_compile_lstm_model(input_shape, num_classes):
    """
    Builds and compiles an LSTM model.
    """
    start_time = time.time()
    print("Building and compiling LSTM model...")

   model = Sequential()
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.3))
    
    # Add an intermediate Dense layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))
    
    model.add(LSTM(128, return_sequences=True))
    
    # Add another intermediate Dense layer with 64 units and ReLU activation
    model.add(Dense(64, activation='relu'))
    
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    
    # Add a final intermediate Dense layer with 32 units and ReLU activation
    model.add(Dense(32, activation='relu'))
    
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
        
    print("LSTM model built and compiled in {:.2f} seconds".format(time.time() - start_time))
    return model

def build_and_compile_bidirectional_lstm_model(input_shape, num_classes):
    """
    Builds and compiles a Bidirectional LSTM model.
    """
    start_time = time.time()
    print("Building and compiling Bidirectional LSTM model...")

    model = Sequential()
    model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=.001), metrics=['accuracy'])
    
    print("Bidirectional LSTM model built and compiled in {:.2f} seconds".format(time.time() - start_time))
    return model

def train_and_evaluate(model, X_train, y_train, X_val, y_val, model_save_path, model_name):
    """
    Trains the model and evaluates its performance.
    """
    start_time = time.time()
    print(f"Training {model_name}...")
    
    checkpoint = ModelCheckpoint(model_save_path, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1)
    model.fit(X_train, y_train, batch_size=16, epochs=200, validation_data=(X_val, y_val), callbacks=[checkpoint])
    model.load_weights(model_save_path)
    loss, accuracy = model.evaluate(X_val, y_val)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
    y_pred = model.predict(X_val)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_val, axis=1)
    report = classification_report(y_true_classes, y_pred_classes, target_names=le.classes_)
    print(report)

    print(f"{model_name} trained and evaluated in {time.time() - start_time:.2f} seconds")


if __name__ == "__main__":
    # File paths
    audio_base_path_train = 'D:/MusicGenreClassification/ismir04_genre/audio/training/'
    metadata_csv_path_train = 'D:/MusicGenreClassification/ismir04_genre/metadata/training/tracklistwav.csv'
    audio_base_path_val = 'D:/MusicGenreClassification/ismir04_genre/audio/development/'
    metadata_csv_path_val = 'D:/MusicGenreClassification/ismir04_genre/metadata/development/tracklistwav.csv'
    model_save_path_lstm = 'D:/MusicGenreClassification/models/lstm_model.h5'
    model_save_path_bi_lstm = 'D:/MusicGenreClassification/models/bilstm_model.h5'

   # Preprocessing training data
    features_train, labels_train = preprocess_data(audio_base_path_train, metadata_csv_path_train)
    le = LabelEncoder()
    labels_train_encoded = to_categorical(le.fit_transform(labels_train))

    # Scale features for RandomForest feature importance analysis
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)

    # Perform feature importance analysis
    feature_importance_df = analyze_feature_importance(features_train_scaled, le.transform(labels_train))

    # Reshape features for LSTM model
    features_train_reshaped = np.reshape(features_train, (features_train.shape[0], 1, features_train.shape[1]))
    features_train_reshaped_scaled = scaler.transform(features_train_reshaped.reshape(features_train_reshaped.shape[0], -1)).reshape(features_train_reshaped.shape)
    
    # Split training data
    X_train, X_val_train, y_train, y_val_train = train_test_split(features_train_reshaped_scaled, labels_train_encoded, test_size=0.3, random_state=42, stratify=labels_train_encoded)

    # Preprocess validation data
    features_val, labels_val = preprocess_data(audio_base_path_val, metadata_csv_path_val)
    labels_val_encoded = to_categorical(le.transform(labels_val))
    features_val_reshaped = np.reshape(features_val, (features_val.shape[0], 1, features_val.shape[1]))
    features_val_reshaped_scaled = scaler.transform(features_val_reshaped.reshape(features_val_reshaped.shape[0], -1)).reshape(features_val_reshaped.shape)

    # Build and compile models
    lstm_model = build_and_compile_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])
    bi_lstm_model = build_and_compile_bidirectional_lstm_model((X_train.shape[1], X_train.shape[2]), y_train.shape[1])

    # Train and evaluate models
    train_and_evaluate(lstm_model, X_train, y_train, X_val_train, y_val_train, model_save_path_lstm, "LSTM Model")
    train_and_evaluate(bi_lstm_model, X_train, y_train, X_val_train, y_val_train, model_save_path_bi_lstm, "Bidirectional LSTM Model")

    # Evaluate on separate validation set
    print("Evaluating LSTM Model on separate validation set...")
    lstm_model.load_weights(model_save_path_lstm)
    loss, accuracy = lstm_model.evaluate(features_val_reshaped_scaled, labels_val_encoded)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')

    print("Evaluating Bidirectional LSTM Model on separate validation set...")
    bi_lstm_model.load_weights(model_save_path_bi_lstm)
    loss, accuracy = bi_lstm_model.evaluate(features_val_reshaped_scaled, labels_val_encoded)
    print(f'Validation Loss: {loss:.4f}')
    print(f'Validation Accuracy: {accuracy:.4f}')
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

# Create Spectrogram
def create_spectrogram(audio_path, save_path):
    y, sr = librosa.load(audio_path)
    plt.figure(figsize=(10, 4))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Function to build and compile the CNN model
def build_and_compile_2DCNN(input_shape_spec, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape_spec),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load images and labels
def load_data(data_directory, genres, target_size=(64, 64)):
    images, labels = [], []
    for root, dirs, files in os.walk(data_directory):
        for file in files:
            if file.lower().endswith('.png'):
                img_path = os.path.join(root, file)
                genre = os.path.basename(root)  # Assuming the genre is the name of the parent directory
                img = load_img(img_path, target_size=target_size, color_mode='rgb')
                img_array = img_to_array(img)
                images.append(img_array)
                labels.append(genre)
    return np.array(images), np.array(labels)

# Paths
base_directory = "D:/MusicGenreClassification/ismir04_genre/audio/training"
save_directory = "D:/MusicGenreClassification/spectrograms"
genres = ['classical', 'electronic', 'jazz', 'metal', 'pop', 'punk', 'rock', 'world']

# Creating spectrograms for all audio files
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.lower().endswith('.wav'):
            audio_path = os.path.join(root, file)
            genre = os.path.basename(os.path.dirname(root))  # Assuming this is the genre
            genre_save_directory = os.path.join(save_directory, genre)
            if not os.path.exists(genre_save_directory):
                os.makedirs(genre_save_directory)
            save_path = os.path.join(genre_save_directory, os.path.splitext(file)[0] + '.png')
            create_spectrogram(audio_path, save_path)
            print(f"Spectrogram saved for {file}")

# Load data
X, y = load_data(save_directory, genres)

if X.size == 0:
    raise ValueError("The feature array X is empty. Check the data loading process and file paths.")
if y.size == 0:
    raise ValueError("The label array y is empty. Check the data loading process and label assignment.")

# Preprocess data
X = X / 255.0
le = LabelEncoder()
y = le.fit_transform(y)
y = to_categorical(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and compile the model
input_shape = X_train.shape[1:]
num_classes = y_train.shape[1]
model = build_and_compile_2DCNN(input_shape, num_classes)

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)

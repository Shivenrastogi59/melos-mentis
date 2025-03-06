import sys
import os

# Add the backend directory to sys.path **before** other imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../backend")))

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pickle
from utils.preprocess import preprocess_data  # Remove "backend." since it's now in sys.path

# Define dataset path
dataset_path = ("./dataset/emotions").replace("\\", "/")

# Load preprocessed data
(train_images, train_labels), (test_images, test_labels) = preprocess_data(dataset_path)

# Convert labels to numerical format
unique_labels = list(set(train_labels))
label_to_index = {label: i for i, label in enumerate(unique_labels)}

train_labels = np.array([label_to_index[label] for label in train_labels])
test_labels = np.array([label_to_index[label] for label in test_labels])

# Convert to one-hot encoding
num_classes = len(unique_labels)
train_labels = keras.utils.to_categorical(train_labels, num_classes)
test_labels = keras.utils.to_categorical(test_labels, num_classes)

# Define CNN model
def build_emotion_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

# Build the model
model = build_emotion_model()

# Train the model
print("Training the emotion detection model...")
history = model.fit(
    train_images[..., np.newaxis], train_labels,
    epochs=25, batch_size=64,
    validation_data=(test_images[..., np.newaxis], test_labels)
)

# Save the trained model as a pickle file
model_filename ="../backend/models/emotion_model.pkl"
with open(model_filename, "wb") as file:
    pickle.dump(model, file)

print(f"Model training complete! Saved as {model_filename}")

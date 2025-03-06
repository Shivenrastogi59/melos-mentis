import pickle
import numpy as np
import cv2
from flask import Blueprint, request, jsonify

# Define the blueprint for this route
emotion_bp = Blueprint("emotion", __name__)

# Load the trained model
model_path = "./models/emotion_model.pkl"
with open(model_path, "rb") as file:
    model = pickle.load(file)

# Define emotion labels
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

def preprocess_image(image_path):
    """Load and preprocess image for emotion prediction."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = np.expand_dims(image, axis=[0, -1])  # Add batch and channel dimensions
    return image / 255.0  # Normalize

@emotion_bp.route("/detect_emotion", methods=["POST"])
def detect_emotion():
    """Endpoint to detect emotions from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    image_path = f"./temp/{file.filename}"  # Temporary storage
    file.save(image_path)

    # Process image & predict emotion
    image = preprocess_image(image_path)
    prediction = model.predict(image)
    emotion_index = np.argmax(prediction)
    
    return jsonify({"emotion": emotion_labels[emotion_index]})


import pickle
import numpy as np
from tensorflow import keras

# Load the trained model
model_filename = "../backend/models/emotion_model.pkl"
with open(model_filename, "rb") as file:
    model = pickle.load(file)

# Print model summary
model.summary()

# Dummy input to check prediction
dummy_input = np.random.rand(1, 48, 48, 1)  # Random test image
prediction = model.predict(dummy_input)

print("Model test prediction:", prediction)

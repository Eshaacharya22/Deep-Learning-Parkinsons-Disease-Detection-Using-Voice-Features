import gradio as gr
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# ---------------- LOAD MODEL ----------------
MODEL_PATH = "Spiral & wave.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 128  # change ONLY if your model used a different size

# ---------------- PREDICTION FUNCTION ----------------
def predict_image(image):
    if image is None:
        return "No image uploaded"

    # Convert to grayscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))

    # Normalize
    image = image / 255.0

    # Reshape for model
    image = image.reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # Predict
    prediction = model.predict(image)[0][0]

    if prediction > 0.5:
        return "Parkinson’s Detected"
    else:
        return "Healthy"

# ---------------- GRADIO UI ----------------
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Spiral / Wave Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Parkinson’s Disease Detection",
    description="Upload a spiral or wave drawing image to detect Parkinson’s disease"
)

# ---------------- RUN ----------------
if __name__ == "__main__":
    interface.launch()

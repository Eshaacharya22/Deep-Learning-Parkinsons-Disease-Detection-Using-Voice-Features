import numpy as np
import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model (keep .h5 file in SAME folder)
MODEL_PATH = "Spiral & wave.h5"
model = load_model(MODEL_PATH)

IMG_SIZE = 128  # change if your model uses different size


def preprocess_image(img):
    img = img.convert("L")              # convert to grayscale
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = img.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    return img


def predict(image):
    img = preprocess_image(image)
    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        return "Parkinson’s Detected"
    else:
        return "Healthy"


# Gradio Interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload Spiral/Wave Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Parkinson’s Disease Detection",
    description="Upload Spiral or Wave drawing image to detect Parkinson’s disease."
)

if __name__ == "__main__":
    interface.launch()

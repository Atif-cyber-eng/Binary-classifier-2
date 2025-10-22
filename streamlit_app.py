import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf

MODEL_PATH = "fruits_cnn.h5"
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model(path):
    return tf.keras.models.load_model(path)

def preprocess_image(image, target_size):
    image = image.convert("RGB")
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    arr = np.asarray(image) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

st.title("🍎 Apple vs 🥭 Mango Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_column_width=True)
    arr = preprocess_image(image, IMG_SIZE)
    model = load_model(MODEL_PATH)
    pred = model.predict(arr)[0][0]
    label = "Mango 🥭" if pred >= 0.5 else "Apple 🍎"
    confidence = pred if pred >= 0.5 else 1 - pred
    st.write(f"**Prediction:** {label} ({confidence*100:.2f}% confidence)")

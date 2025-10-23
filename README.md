# 🍎 Apple vs 🥭 Mango Classifier

This is a simple web app built with **Streamlit** that classifies uploaded images as either **Apple** or **Mango** using a **CNN model**.

---

## Features

- Upload an image in JPG, JPEG, or PNG format.
- Preprocesses the image to the required size (224x224).
- Classifies the image as Apple 🍎 or Mango 🥭.
- Displays the prediction with confidence percentage.
- Model is downloaded from Google Drive if not already present.

---

## Requirements

- Python 3.10+
- Streamlit
- TensorFlow
- Pillow
- NumPy
- gdown

Install dependencies using:

```bash
pip install -r requirements.txt

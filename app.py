import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load model
model_path = "path of the almond photo"
model = tf.keras.models.load_model(model_path)

# Class names (adjust as per your training order)
class_names = ['AK', 'KAPADOKYA', 'NURLU', 'SIRA']

# Streamlit UI
st.title("🌰 Almond Variety Classifier")
st.write("Upload an almond image to predict its variety")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocessing
    img = img.resize((128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    st.success(f"🔍 Predicted Almond Variety: **{predicted_class}**")

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

# Load your custom model
model = keras.models.load_model('custom_face_model.keras')

# Set the names according to your model's output
CLASS_NAMES = ['Asadbek', 'Temurbek']

st.title("Asadbek va Temurbekni Tanib Olish")

# Function to predict the face
def predict_face(image):
    # Preprocess the image to match your model's expected input
    img = cv2.resize(image, (224, 224))  # Adjust size based on your model
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Make prediction
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0]) * 100
    
    return CLASS_NAMES[predicted_class], confidence

# Camera input
picture = st.camera_input("Suratga olish uchun kamera tugmasini bosing")

if picture:
    # Convert the image to OpenCV format
    image = Image.open(picture)
    image = np.array(image)
    
    # Convert RGB to BGR (OpenCV uses BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Display the captured image
    st.image(picture, caption="Olingan surat", use_column_width=True)
    
    # Make prediction
    name, confidence = predict_face(image)
    
    # Show results
    st.success(f"Tanilgan shaxs: {name}")
    st.info(f"Isbot: {confidence:.2f}%")
    
    # Add some fun elements
    if confidence > 90:
        st.balloons()
    elif confidence < 70:
        st.warning("Past aniqlik darajasi. Yana bir bor urinib ko'ring.")

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tempfile
import time

# Load the pre-trained model
model = load_model('custom_face_model.keras')

# Define class labels
labels = {0: "Asadbek", 1: "Temurbek"}

# Function to preprocess the captured image
def preprocess_image(image):
    # Resize to match model input (assuming 224x224, adjust if different)
    img = cv2.resize(image, (224, 224))
    # Convert to RGB (if model expects RGB)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize pixel values
    img = img / 255.0
    # Expand dimensions to match model input
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit app
st.title("Face Recognition with Webcam")

# Instructions
st.write("Click the button below to capture an image from your webcam.")

# Placeholder for webcam feed
video_placeholder = st.empty()

# Button to capture image
if st.button("Capture Image"):
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not access webcam.")
    else:
        # Display webcam feed
        st.write("Webcam is active. Press 'Capture Image' again to take a photo.")
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Display live feed
                video_placeholder.image(frame, channels="BGR")
                # Check for capture button press (simulated by checking Streamlit's state)
                if st.session_state.get("capture", False):
                    break
                # Simulate capture on second button press
                if st.button("Confirm Capture"):
                    st.session_state.capture = True
                    break
            time.sleep(0.1)  # Control frame rate

        if ret:
            # Preprocess the captured frame
            processed_img = preprocess_image(frame)
            # Make prediction
            prediction = model.predict(processed_img)
            # Get class and confidence
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100
            predicted_name = labels[predicted_class]
            # Display results
            st.image(frame, channels="BGR", caption="Captured Image")
            st.write(f"Predicted: **{predicted_name}** with **{confidence:.2f}%** confidence")
        else:
            st.error("Error: Could not capture image.")
        # Release webcam
        cap.release()
else:
    st.write("Webcam not active. Click 'Capture Image' to start.")

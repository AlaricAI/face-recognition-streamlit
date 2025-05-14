import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🤖 Real-Time Face Recognition")
st.markdown("Kameraga qarang va shaxsingiz aniqlansin!")

# 🔍 Modelni yuklash
model = load_model("custom_face_model.keras")  # Modelni Streamlit va GitHub'da ishlayotgan joyga qarab o'zgartiring

class_names = ["Asadbek", "Temurbek"]  # Moslab yoz

# 🧠 Predict funktsiyasi
def predict_face(face_img):
    face_resized = cv2.resize(face_img, (224, 224))  # Model inputiga qarab
    if face_resized.shape[-1] == 4:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_RGBA2RGB)
    elif face_resized.shape[-1] == 1:
        face_resized = cv2.cvtColor(face_resized, cv2.COLOR_GRAY2RGB)

    face_array = face_resized.astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)
    prediction = model.predict(face_array)[0]
    best_idx = np.argmax(prediction)
    name = class_names[best_idx]
    confidence = prediction[best_idx] * 100
    return name, confidence

# 📷 Kamera orqali rasm olish
img_file_buffer = st.camera_input("📸 Suratga tushish uchun kamerani yoqing")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)

    st.image(img_array, caption="Olingan rasm", use_container_width=True)

    # Haar kaskad bilan yuzni aniqlash
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        st.warning("🙈 Yuz aniqlanmadi.")
    else:
        for (x, y, w, h) in faces:
            face_img = img_array[y:y+h, x:x+w]
            name, confidence = predict_face(face_img)

            st.success(f"👤 Tanildi: **{name}** — {confidence:.2f}% ishonch bilan.")

            # Vizual ko‘rsatish uchun yuz atrofiga chiziq
            img_array = cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(img_array, caption="Yuzlar belgilandi", use_container_width=True)

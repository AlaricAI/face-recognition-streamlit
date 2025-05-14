import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.models import load_model

st.title("Real-Time Face Recognition")

# Kamera ochish
camera = cv2.VideoCapture(2)

stframe = st.empty()

# Modelni yuklash
model = load_model("custom_face_model.keras")
class_names = ["Asadbek", "Temurbek"]  # bu yerda modelga mos classlar bo'lishi kerak

def predict_face(face_img):
    resized = cv2.resize(face_img, (224, 224))  # model inputiga qarab o'lcham o'zgartir
    img_array = resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    best_idx = np.argmax(prediction)
    name = class_names[best_idx]
    confidence = prediction[best_idx] * 100
    return name, confidence

st.markdown("## 📷 Rasmga olish uchun 'Rasmga olish' tugmasini bosing")

if st.button("📸 Rasmga olish"):
    ret, frame = camera.read()
    if not ret:
        st.error("Kamera orqali rasm olinmadi.")
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        st.image(rgb, caption="Olingan rasm", use_column_width=True)

        # Yuzni aniqlash (oddiy haarcascade usuli, istasa dlib yoki MTCNN ishlatish mumkin)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(rgb, 1.1, 4)

        if len(faces) == 0:
            st.warning("Yuz topilmadi.")
        else:
            for (x, y, w, h) in faces:
                face_img = rgb[y:y+h, x:x+w]
                name, confidence = predict_face(face_img)
                st.success(f"👤 Aniqlangan: **{name}** ({confidence:.2f}% aniqlik)")

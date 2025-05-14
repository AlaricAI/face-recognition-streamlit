import streamlit as st
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
from keras.models import load_model

st.set_page_config(page_title="Face Recognition", layout="centered")
st.title("🤖 Real-Time Face Recognition")
st.markdown("Suratga tush va biz seni kimligingni aniqlaymiz!")

# 📦 Modelni yuklash
model = load_model("custom_face_model.keras")
class_names = ["Asadbek", "Temurbek"]  # O'zingga moslab o'zgartir

# 🔍 Modelga input tayyorlash funksiyasi
def predict_face(face_img):
    resized = cv2.resize(face_img, (224, 224))  # model inputiga qarab
    img_array = resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)[0]
    best_idx = np.argmax(prediction)
    name = class_names[best_idx]
    confidence = prediction[best_idx] * 100
    return name, confidence

# 📷 Kamera input (web uchun)
img_file_buffer = st.camera_input("📸 Suratga tushish uchun kamerani yoqing")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer)
    img_array = np.array(img)
    st.image(img_array, caption="Olingan rasm", use_column_width=True)

    # Yuzni aniqlash
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(img_array, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("🙈 Yuz topilmadi.")
    else:
        for (x, y, w, h) in faces:
            face_img = img_array[y:y+h, x:x+w]
            name, confidence = predict_face(face_img)
            st.success(f"👤 Aniqlangan: **{name}** ({confidence:.2f}% ishonch)")

            # Yuzni chizib ko‘rsatish
            cv2.rectangle(img_array, (x, y), (x + w, y + h), (0, 255, 0), 2)

        st.image(img_array, caption="Yuz belgilangan rasm", use_column_width=True)

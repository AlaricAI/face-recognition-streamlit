import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Modelni yuklash
model = load_model("custom_face_model.keras")

# Class nomlari (o'zing to'g'irla kerak bo'lsa)
class_names = ["Asadbek", "Temurbek"]

st.title("Yuzni aniqlash: Asadbekmi yoki Temurbek?")

# Kamera ochiladi
camera = st.camera_input("Rasmga olish uchun kamerani yoqing:")

if camera:
    # Rasmni ochish
    img = Image.open(camera)
    img_array = np.array(img)

    # OpenCV RGB -> BGR aylantirish (ba'zi modellar uchun kerak bo'lishi mumkin)
    img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Modelga mos qilish (o'lcham va normallash)
    resized = cv2.resize(img_array, (224, 224))  # modelga mos bo'lgan o'lchamni yoz!
    normalized = resized / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    # Bashorat qilish
    predictions = model.predict(input_tensor)[0]
    predicted_index = np.argmax(predictions)
    confidence = predictions[predicted_index]

    # Natijani chiqarish
    st.subheader(f"Tanilgan shaxs: **{class_names[predicted_index]}**")
    st.write(f"Aniqlik darajasi: **{confidence * 100:.2f}%**")

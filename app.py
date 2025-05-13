import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf

# Modelni yuklash
model = load_model('custom_face_model.keras')

# Klass nomlari (Asadbek va Temurbek)
class_names = ['Asadbek', 'Temurbek']

# Yuzni aniqlash uchun Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuz tasvirini model uchun tayyorlash
def preprocess_image(image):
    # Yuzni aniqlash
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None
    
    # Birinchi aniqlangan yuzni olish
    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    
    # Yuzni model uchun o'lchamini o'zgartirish (modelingizga mos ravishda o'zgartiring)
    face = cv2.resize(face, (224, 224))  # Model o'lchamiga moslashtiring
    face = face / 255.0  # Normalizatsiya
    face = np.expand_dims(face, axis=0)  # Batch o'lchamini qo'shish
    
    return face, (x, y, w, h)

# Streamlit ilovasi
st.title("Yuzni Aniqlash Ilovasi")
st.write("Kamera orqali Asadbek yoki Temurbekni aniqlash")

# Kamera tasvirini olish uchun Streamlit komponenti
run = st.checkbox('Kamerani ishga tushirish')
FRAME_WINDOW = st.image([])

cap = cv2.VideoCapture(0)  # Veb-kamerani ochish

if not cap.isOpened():
    st.error("Kamera ochilmadi. Iltimos, kamerangiz ulanganligini tekshiring.")
else:
    while run:
        ret, frame = cap.read()
        if not ret:
            st.error("Kameradan tasvir olinmadi.")
            break
        
        # Tasvirni RGB formatiga o'tkazish (Streamlit uchun)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Yuzni aniqlash va modelga tayyorlash
        processed_face, face_coords = preprocess_image(frame_rgb)
        
        if processed_face is not None:
            # Model orqali bashorat qilish
            predictions = model.predict(processed_face)
            score = tf.nn.softmax(predictions[0])
            predicted_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
            # Natijani tasvirga chizish
            if face_coords is not None:
                (x, y, w, h) = face_coords
                cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                text = f"{predicted_class}: {confidence:.2f}%"
                cv2.putText(frame_rgb, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Natijani Streamlit'da ko'rsatish
            st.write(f"Aniqlangan shaxs: **{predicted_class}** ({confidence:.2f}%)")
        
        # Tasvirni Streamlit'da ko'rsatish
        FRAME_WINDOW.image(frame_rgb)
        
        # Kamerani to'xtatish uchun
        if not run:
            break
    
    cap.release()

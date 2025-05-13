import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Modelni yuklash
model = load_model('custom_face_model.keras')

# Klass nomlari
class_names = ['Asadbek', 'Temurbek']

# Yuzni aniqlash uchun Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuz tasvirini model uchun tayyorlash
def preprocess_image(image):
    # PIL tasvirini OpenCV formatiga o'tkazish
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Yuzni aniqlash
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None, None, image
    
    # Birinchi aniqlangan yuzni olish
    (x, y, w, h) = faces[0]
    face = image[y:y+h, x:x+w]
    
    # Yuzni model uchun o'lchamini o'zgartirish
    face = cv2.resize(face, (224, 224))  # Model o'lchamiga moslashtiring
    face = face / 255.0  # Normalizatsiya
    face = np.expand_dims(face, axis=0)  # Batch o'lchamini qo'shish
    
    return face, (x, y, w, h), image

# Streamlit ilovasi
st.title("Yuzni Aniqlash Ilovasi")
st.write("Kamera orqali Asadbek yoki Temurbekni aniqlash")

# Brauzer kamerasidan tasvir olish
camera_image = st.camera_input("Kamerani yoqing")

if camera_image is not None:
    # Tasvirni o'qish
    image = Image.open(camera_image)
    
    # Yuzni aniqlash va tayyorlash
    processed_face, face_coords, original_image = preprocess_image(image)
    
    if processed_face is not None:
        # Model orqali bashorat qilish
        predictions = model.predict(processed_face)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)
        
        # Natijani tasvirga chizish
        original_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        (x, y, w, h) = face_coords
        cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{predicted_class}: {confidence:.2f}%"
        cv2.putText(original_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Tasvirni RGB formatiga qaytarish va ko'rsatish
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        st.image(original_image, caption="Aniqlangan yuz")
        st.write(f"Aniqlangan shaxs: **{predicted_class}** ({confidence:.2f}%)")
    else:
        st.error("Yuz aniqlanmadi. Iltimos, yuzingizni kamera oldida aniq ko'rsating.")

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import cv2
import tensorflow as tf

# Modelni yuklash
@st.cache_resource
def load_my_model():
    try:
        model_path = 'custom_face_model.keras'
        model = keras_load_model(model_path)
        st.success("Model muvaffaqiyatli yuklandi!")
        return model
    except Exception as e:
        st.error(f"Model yuklanmadi. Xato: {str(e)}")
        return None

model = load_my_model()

# Yuzni aniqlash uchun Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Model kategoriyalari - BU QISMI MODEL O'QITILGAN TARTIBGA QAT'IY MOS KELISHI KERAK!
# Agar model 0-indeks Asadbek, 1-indeks Temurbek deb o'qitilgan bo'lsa:
CATEGORIES = ['Temurbek', 'Asadbek']  # BU TARTIBNI MODEL O'QITISHDA ISHLATILGAN TARTIBGA MOSLASHTIRING

# Yuzni aniqlash va model uchun tayyorlash funksiyasi
def predict_face(image):
    try:
        # PIL tasvirini OpenCV formatiga o'tkazish
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Yuzni aniqlash (parametrlarni yumshatish)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))
        
        if len(faces) == 0:
            return None, None, img, "Yuz aniqlanmadi: Iltimos, yuzingizni kameraga yaqinroq tuting yoki yorug'likni yaxshilang."
        
        # Birinchi aniqlangan yuzni olish
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        
        # Yuzni model uchun tayyorlash
        face = cv2.resize(face, (50, 37))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)
        
        # Model yordamida bashorat qilish
        pred = model.predict(face)
        pred = tf.nn.softmax(pred[0]).numpy()
        return pred, (x, y, w, h), img, None
    except Exception as e:
        return None, None, img, f"Yuzni tahlil qilishda xato: {str(e)}"

# Streamlit ilovasi
st.title('Yuzni Aniqlash Modeli')
st.write("Kameradan rasm oling, model shaxsning ismini (Asadbek yoki Temurbek) aniqlaydi.")

# Qo'llanma
st.sidebar.header("Ko'rsatmalar")
st.sidebar.write("""
1. Kamerani ishga tushiring va yuzni aniq ko'rinadigan rasm oling.
2. Yuzingizni kameraga yaqin tuting va yaxshi yoritilgan joyda turing.
3. Model shaxsning ismini aniqlaydi va ishonchlilik foizini ko'rsatadi.
Eslatma: Yaxshiroq natija uchun yuzingizni to'g'ridan-to'g'ri kameraga qarating.
""")

# Kameradan rasm olish uchun tugma
video_file = st.camera_input("Kamera bilan rasm oling")

if video_file is not None and model is not None:
    try:
        # Rasmdan img ni oling
        img = Image.open(video_file)
        st.image(img, caption="Kamera orqali olingan rasm", width=300)

        # Model yordamida bashorat qilish
        pred, face_coords, original_image, error = predict_face(img)

        if error:
            st.error(error)
        elif pred is not None:
            # Eng yuqori ehtimollikdagi kategoriyani aniqlash
            predicted_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_name = CATEGORIES[predicted_index]

            # Debug ma'lumotlari
            st.write(f"Model chiqishlari: {CATEGORIES[1]}: {pred[1]*100:.1f}%, {CATEGORIES[0]}: {pred[0]*100:.1f}%")
            st.write(f"Tanlangan indeks: {predicted_index}")

            # Yuzni ramkaga olish
            (x, y, w, h) = face_coords
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{predicted_name}: {confidence:.1f}%"
            cv2.putText(original_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tasvirni ko'rsatish
            st.image(original_image, caption="Aniqlangan yuz", width=300)

            # Natijalarni ko'rsatish
            st.subheader("Asosiy natija:")
            st.metric(label="Ism", value=predicted_name)
            st.write(f"Ishonchlilik darajasi: {confidence:.1f}%")

    except Exception as e:
        st.error(f"Rasmni tahlil qilishda xato: {str(e)}")

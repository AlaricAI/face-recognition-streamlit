import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import cv2

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

# Yuzni aniqlash va model uchun tayyorlash funksiyasi
def predict_face(image):
    try:
        # PIL tasvirini OpenCV formatiga o'tkazish
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Yuzni aniqlash
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) == 0:
            return None, None, img, "Yuz aniqlanmadi"
        
        # Birinchi aniqlangan yuzni olish
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]
        
        # Yuzni model uchun tayyorlash
        face = cv2.resize(face, (50, 37))  # Model o'lchamiga moslashtiring
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)  # Grayscale ga o'tkazish
        face = face / 255.0  # Normalizatsiya
        face = np.expand_dims(face, axis=-1)  # (50, 37, 1) shaklini yaratish
        face = np.expand_dims(face, axis=0)  # Batch o'lchamini qo'shish
        
        # Model yordamida bashorat qilish
        pred = model.predict(face)
        return pred, (x, y, w, h), img, None
    except Exception as e:
        return None, None, img, f"Yuzni tahlil qilishda xato: {str(e)}"

# Streamlit ilovasi
st.title('Yuzni Aniqlash Modeli')
st.write("Kameradan rasm oling, model shaxsning ismini (Asadbek yoki Temurbek) aniqlaydi.")

# Qo‘llanma
st.sidebar.header("Ko‘rsatmalar")
st.sidebar.write("""
1. Kamerani ishga tushiring va yuzni aniq ko‘rinadigan rasm oling.
2. Model shaxsning ismini aniqlaydi.
3. Natijalar ishonchlilik foizi bilan ko‘rsatiladi.

Eslatma: Yaxshiroq natija uchun faqat yuz ko‘rinadigan rasmlardan foydalaning.
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
            predicted_class = np.argmax(pred[0])
            categories = ['Asadbek', 'Temurbek']  # Tartibni test orqali aniqlang
            predicted_name = categories[predicted_class]
            confidence = pred[0][predicted_class] * 100

            # Yuzni ramkaga olish
            (x, y, w, h) = face_coords
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"{predicted_name}: {confidence:.1f}%"
            cv2.putText(original_image, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Tasvirni ko‘rsatish
            st.image(original_image, caption="Aniqlangan yuz", width=300)

            # Natijalarni ko‘rsatish
            st.subheader("Asosiy natija:")
            st.metric(label="Ism", value=predicted_name)
            st.write(f"Ishonchlilik darajasi: {confidence:.1f}%")

            # Ehtimolliklar grafigi
            st.subheader("Barcha kategoriyalar bo‘yicha ehtimollar:")
            df = pd.DataFrame({
                'Kategoriya': ['Asadbek', 'Temurbek'],
                'Ehtimollik (%)': [pred[0][0] * 100, pred[0][1] * 100]
            })

            # Saralash
            df = df.sort_values('Ehtimollik (%)', ascending=False)

            # Plotly bar chart
            fig = px.bar(df, 
                         x='Kategoriya', 
                         y='Ehtimollik (%)',
                         color='Ehtimollik (%)',
                         color_continuous_scale='Bluered',
                         text='Ehtimollik (%)',
                         title='Yuzni aniqlash ehtimollari')
            
            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_yaxes(range=[0, 100])  # 0-100% oralig‘ini ko‘rsatish
            
            st.plotly_chart(fig, use_container_width=True)

            # Pie chart
            fig_pie = px.pie(df,
                             names='Kategoriya',
                             values='Ehtimollik (%)',
                             title='Ehtimollarning taqsimlanishi')
            
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error("Yuz aniqlanmadi. Iltimos, yuzingizni kamera oldida aniq ko‘rsating.")
    except Exception as e:
        st.error(f"Rasmni tahlil qilishda xato: {str(e)}")

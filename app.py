import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import io
import base64

# Modelni yuklash
@st.cache_resource
def load_my_model():
    try:
        model_path = 'custom_face_model.keras'
        return keras_load_model(model_path)
    except Exception as e:
        st.error(f"Model yuklanmadi. Xato: {str(e)}")
        return None

model = load_my_model()

# Yuzni aniqlash funksiyasi
def predict_face(image):
    # Rasmni tayyorlash
    img = image.resize((50, 37))  # Modelning kiritish shakliga moslashtirish
    img = img.convert('L')  # Grayscale ga o'tkazish
    img = np.array(img) / 255.0  # Normalizatsiya
    img = np.expand_dims(img, axis=-1)  # (50, 37, 1) shaklini yaratish
    img = np.expand_dims(img, axis=0)  # Batch o'lchamini qo'shish
    pred = model.predict(img)
    return pred

# Kameradan rasm olish
st.title('Kamera orqali yuzni tanish')

# HTML video orqali kamerani ko'rsatish
st.write('Kameradan rasm olish uchun quyidagi tugmani bosing:')
video_file = st.camera_input("Kamera bilan rasm oling")

if video_file:
    try:
        # Rasmdan img ni oling
        img = Image.open(video_file)
        st.image(img, caption="Kamera orqali rasm", use_column_width=True)

        # Model yordamida bashorat qilish
        if model is None:
            st.error("Model yuklanmagan. Iltimos, model faylini tekshiring.")
        else:
            pred = predict_face(img)

            # Natijani ko'rsatish
            st.write(f"Model natijasi: {pred}")

            # Eng yuqori ehtimollikdagi kategoriyani aniqlash
            predicted_class = np.argmax(pred[0])
            categories = ['Asadbek', 'Temurbek']
            predicted_name = categories[predicted_class]
            confidence = pred[0][predicted_class] * 100

            st.subheader(f"Tahmin qilingan ism: {predicted_name} ({confidence:.2f}% ehtimollik bilan)")

            # Natijalar ko'rsatiladi
            st.subheader("Ehtimolliklar:")
            df = pd.DataFrame({
                'Kategoriya': ['Asadbek', 'Temurbek'],
                'Ehtimollik (%)': [pred[0][0] * 100, pred[0][1] * 100]
            })

            fig = px.bar(df, x='Kategoriya', y='Ehtimollik (%)', color='Ehtimollik (%)',
                         title="Yuzni aniqlash ehtimollari")
            st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Xatolik yuz berdi: {str(e)}")

# Qo'shimcha ma'lumotlar
st.sidebar.header("Qo'llanma")
st.sidebar.write("""
1. Kamerani ishga tushiring va rasm oling.
2. Model rasmni tahlil qiladi va Asadbek yoki Temurbek ekanligini aniqlaydi.
""")

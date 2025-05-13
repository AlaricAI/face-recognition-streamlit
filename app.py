import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image

# Modelni yuklash
@st.cache_resource
def load_my_model():
    try:
        model_path = 'custom_face_model.keras'
        model = keras_load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Model yuklanmadi. Xato: {str(e)}")
        return None

model = load_my_model()

# Yuzni aniqlash funksiyasi
def predict_face(image):
    # Rasmni tayyorlash
    img = image.resize((50, 37))  # Modelning kiritish shakliga moslashtirish
    img = img.convert('L')  # Grayscale ga o‘tkazish
    img = np.array(img) / 255.0  # Normalizatsiya
    img = np.expand_dims(img, axis=-1)  # (50, 37, 1) shaklini yaratish
    img = np.expand_dims(img, axis=0)  # Batch o‘lchamini qo‘shish
    pred = model.predict(img)
    return pred

# Kameradan rasm olish
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
        pred = predict_face(img)

        # Eng yuqori ehtimollikdagi kategoriyani aniqlash
        predicted_class = np.argmax(pred[0])
        categories = ['Asadbek', 'Temurbek']  # Tartibni test orqali aniqlang
        predicted_name = categories[predicted_class]
        confidence = pred[0][predicted_class] * 100

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

    except Exception as e:
        st.error(f"Rasmni tahlil qilishda xato: {str(e)}")

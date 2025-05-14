import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from tensorflow.keras.models import load_model as keras_load_model
from PIL import Image
import cv2
import tensorflow as tf

# Modelni yuklash
@st.cache_resource  # Agar eski Streamlit bo‘lsa, bu yerga @st.cache(allow_output_mutation=True) deb yozing
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

# Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Yuzni aniqlash va model uchun tayyorlash funksiyasi
def predict_face(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        if len(faces) == 0:
            return None, None, img, "Yuz aniqlanmadi: Iltimos, yuzingizni kameraga yaqinroq tuting yoki yorug'likni yaxshilang."

        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]

        face = cv2.resize(face, (50, 37))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        face = face / 255.0
        face = np.expand_dims(face, axis=-1)
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)
        pred = tf.nn.softmax(pred[0]).numpy()

        return pred, (x, y, w, h), img, None
    except Exception as e:
        return None, None, img, f"Yuzni tahlil qilishda xato: {str(e)}"

# Streamlit interfeysi
st.title('Yuzni Aniqlash Modeli')
st.write("Kameradan rasm oling, model shaxsning ismini (Asadbek yoki Temurbek) aniqlaydi.")

st.sidebar.header("Ko‘rsatmalar")
st.sidebar.write("""
1. Kamerani ishga tushiring va yuzni aniq ko‘rinadigan rasm oling.
2. Yuzingizni kameraga yaqin tuting va yaxshi yoritilgan joyda turing.
3. Model shaxsning ismini aniqlaydi va ishonchlilik foizini ko‘rsatadi.
""")

video_file = st.camera_input("Kamera bilan rasm oling")

if video_file is not None and model is not None:
    try:
        img = Image.open(video_file)
        st.image(img, caption="Kamera orqali olingan rasm", width=300)

        pred, face_coords, original_image, error = predict_face(img)

        if error:
            st.error(error)
        elif pred is not None:
            # === MUHIM: Kategoriya tartibini to‘g‘ri belgilang ===
            categories = ['Asadbek', 'Temurbek']  # EHTIYOT BO‘L! Agar kerak bo‘lsa joyini almashtir!

            predicted_class = np.argmax(pred)
            predicted_name = categories[predicted_class]
            confidence = pred[predicted_class] * 100

            # Debug uchun ehtimolliklar
            st.write(f"Xom ehtimolliklar: {categories[0]}: {pred[0]*100:.1f}%, {categories[1]}: {pred[1]*100:.1f}%")

            (x, y, w, h) = face_coords
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(original_image, f"{predicted_name}: {confidence:.1f}%", (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            st.image(original_image, caption="Aniqlangan yuz", width=300)

            st.subheader("Asosiy natija:")
            st.metric(label="Ism", value=predicted_name)
            st.write(f"Ishonchlilik darajasi: {confidence:.1f}%")

            st.subheader("Barcha kategoriyalar bo‘yicha ehtimollar:")
            df = pd.DataFrame({
                'Kategoriya': categories,
                'Ehtimollik (%)': pred * 100
            })

            df = df.sort_values('Ehtimollik (%)', ascending=False)

            fig = px.bar(df, 
                         x='Kategoriya', 
                         y='Ehtimollik (%)',
                         color='Ehtimollik (%)',
                         color_continuous_scale='Bluered',
                         text='Ehtimollik (%)',
                         title='Yuzni aniqlash ehtimollari')

            fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
            fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
            fig.update_yaxes(range=[0, 100])

            st.plotly_chart(fig, use_container_width=True)

            fig_pie = px.pie(df,
                             names='Kategoriya',
                             values='Ehtimollik (%)',
                             title='Ehtimollarning taqsimlanishi')
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.error("Yuz aniqlanmadi. Iltimos, yuzingizni kamera oldida aniq ko‘rsating.")
    except Exception as e:
        st.error(f"Rasmni tahlil qilishda xato: {str(e)}")

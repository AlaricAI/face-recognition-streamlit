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
        st.success("✅ Model muvaffaqiyatli yuklandi!")
        st.write(f"ℹ️ Model kutilayotgan input shakli: {model.input_shape}")
        return model
    except Exception as e:
        st.error(f"❌ Model yuklanmadi. Xato: {str(e)}")
        return None

model = load_my_model()

# Yuz aniqlash uchun HaarCascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# To‘g‘ri tartibda model o‘qitilgan deb hisoblaymiz (A-Z alfavit bo‘yicha)
CATEGORIES = ['Asadbek', 'Temurbek']  # EHTIYOT BO‘L: modelga aynan shu tartibda o‘qitilgan bo‘lishi kerak

# Yuzni aniqlash va model uchun tayyorlash funksiyasi
def predict_face(image):
    try:
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=3, minSize=(20, 20))

        if len(faces) == 0:
            return None, None, img, "⚠️ Yuz aniqlanmadi: Iltimos, yuzingizni kameraga yaqinroq tuting."

        # Eng katta yuzni tanlaymiz
        faces = sorted(faces, key=lambda box: box[2]*box[3], reverse=True)
        (x, y, w, h) = faces[0]
        face = img[y:y+h, x:x+w]

        # Rasmni modelga moslashtirish
        input_shape = model.input_shape[1:3]
        face = cv2.resize(face, (input_shape[1], input_shape[0]))  # (width, height)
        
        if model.input_shape[-1] == 1:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            face = np.expand_dims(face, axis=-1)
        else:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        face = face / 255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face)[0]  # Softmax chiqish bo'lishi kerak
        return pred, (x, y, w, h), img, None
    except Exception as e:
        return None, None, img, f"❌ Tahlil xatosi: {str(e)}"

# Streamlit UI
st.title('🧠 Yuzni Aniqlash Ilovasi')
st.write("📷 Kameradan rasm oling, model esa kimligini aniqlasin (Asadbek yoki Temurbek).")

# Qo'llanma
st.sidebar.header("📝 Qo'llanma")
st.sidebar.write("""
1. Kamera orqali rasm oling.
2. Yuzingiz to‘liq ko‘rinsin.
3. Model sizni tanib oladi.
""")

# Rasm olish
video_file = st.camera_input("📸 Rasm olish")

if video_file is not None and model is not None:
    try:
        img = Image.open(video_file)
        st.image(img, caption="Rasm olindi", width=300)

        pred, face_coords, original_image, error = predict_face(img)

        if error:
            st.error(error)
        elif pred is not None:
            predicted_index = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_name = CATEGORIES[predicted_index]

            st.write(f"🧪 Model natijalari: {CATEGORIES[0]}: {pred[0]*100:.1f}%, {CATEGORIES[1]}: {pred[1]*100:.1f}%")
            st.write(f"📍 Eng ishonchli kategoriya: {predicted_name}")

            # Vizualizatsiya
            if face_coords is not None:
                (x, y, w, h) = face_coords
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                cv2.rectangle(original_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(original_image, f"{predicted_name}: {confidence:.1f}%", 
                            (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                st.image(original_image, caption="📍 Aniqlangan yuz", width=300)

            st.subheader("🔎 Natija")
            st.metric(label="👤 Ism", value=predicted_name)
            st.write(f"Ishonchlilik darajasi: **{confidence:.1f}%**")

            df = pd.DataFrame({
                "Kategoriya": CATEGORIES,
                "Ehtimollik (%)": [pred[0]*100, pred[1]*100]
            })
            fig = px.bar(df, x="Kategoriya", y="Ehtimollik (%)", color="Ehtimollik (%)")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"🚨 Xatolik yuz berdi: {str(e)}")

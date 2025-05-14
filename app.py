import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Modelni yuklash
try:
    model = load_model('custom_face_model.keras')
    st.success("Model muvaffaqiyatli yuklandi!")
except Exception as e:
    st.error(f"Modelni yuklashda xato: {e}")
    st.stop()

# Sinf nomlari
labels = {0: "Asadbek", 1: "Temurbek"}

# Rasmni oldindan ishlov berish funksiyasi
def preprocess_image(image):
    # Rasmni model kiritish o‘lchamiga moslashtirish (224x224 deb faraz qilamiz)
    img = cv2.resize(image, (224, 224))
    # RGB formatiga o‘tkazish
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Piksellarni normallashtirish
    img = img / 255.0
    # Model kiritishi uchun o‘lchamni kengaytirish
    img = np.expand_dims(img, axis=0)
    return img

# Streamlit ilovasi
st.title("Yuzni Aniqlash Ilovasi")
st.write("Kameradan rasm olish uchun quyidagi tugmani bosing.")

# Kamera orqali rasm olish
uploaded_image = st.camera_input("Rasmga olish")

if uploaded_image is not None:
    # Rasmni o‘qish
    bytes_data = uploaded_image.getvalue()
    nparr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Rasmni ko‘rsatish
    st.image(img, channels="BGR", caption="Olingan rasm")
    
    # Rasmni oldindan ishlov berish
    processed_img = preprocess_image(img)
    
    # Bashorat qilish
    try:
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class] * 100
        predicted_name = labels[predicted_class]
        
        # Natijani ko‘rsatish
        st.write(f"Bashorat: **{predicted_name}** ({confidence:.2f}% aniqlik bilan)")
    except Exception as e:
        st.error(f"Bashorat qilishda xato: {e}")
else:
    st.write("Iltimos, rasm olish uchun kameradan foydalaning.")

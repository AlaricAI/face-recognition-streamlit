import streamlit as st
import face_recognition
import numpy as np
from PIL import Image

# Kameradan rasm olish
st.title("Yuzni Tanib Olish")
image = st.camera_input("Rasmni olish uchun kamerani ishga tushiring")

if image is not None:
    # Rasmni PIL formatiga o‘zgartiramiz
    image = Image.open(image)
    image = np.array(image)
    
    # Yuzni aniqlash
    face_locations = face_recognition.face_locations(image)
    
    if face_locations:
        st.image(image, caption="Yuz aniqlangan", use_column_width=True)
        st.write(f"{len(face_locations)} ta yuz topildi!")
    else:
        st.write("Yuz topilmadi.")

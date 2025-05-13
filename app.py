import streamlit as st
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image

model = load_model('custom_face_model.keras')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

st.title("Yuzni Tanlash Tizimi")
run = st.button('Kamerani ishga tushirish')

if run:
    st.write("Kamera ishga tushdi!")
    camera = cv2.VideoCapture(0)
    while True:
        ret, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (37, 50))
            face_array = np.array(face_resized).reshape(1, 50, 37, 1) / 255.0
            
            prediction = model.predict(face_array)
            predicted_label = np.argmax(prediction)
            
            label = labels[predicted_label]
            confidence = np.max(prediction)
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(frame, f'{label}: {confidence:.2f}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        st.image(frame, channels="BGR", use_column_width=True)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()

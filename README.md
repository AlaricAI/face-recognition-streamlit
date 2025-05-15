# 🧠 Face Recognition Streamlit App (WIP)

A Streamlit-based face recognition app that detects **only two people** (for now).  
Still under development — expect bugs, coffee spills, and quantum anomalies.

## ⚙️ Features

- 🧑‍🤝‍🧑 Detects faces of two specific people (hardcoded)
- 📸 Upload or use live camera feed (if supported)
- 🧪 Educational demo for face recognition using OpenCV + face_recognition

## 🚧 Current Limitations

- Detects only two known people
- Accuracy is not reliable yet
- No persistent database or training feature (yet!)

## 🛠️ Installation

```bash
git clone https://github.com/AlaricAI/face-recognition-streamlit
cd face-recognition-streamlit
pip install -r requirements.txt
streamlit run app.py

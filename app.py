import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import gdown
import os

model_path = "Now_the_best_model.keras"
if not os.path.exists(model_path):
    url = "https://drive.google.com/uc?id=1Mmi7UhWsZ4svpFm8TybRfUyBSOJXpJPi"
    gdown.download(url, model_path, quiet=False)

# Page config
st.set_page_config(page_title="Mood Detection", layout="centered")

# Title
st.markdown("<h1 style='text-align: center; color: red;'>😊 Mood Detection App</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_my_model():
    return load_model(model_path)

model = load_my_model()

emotion_labels = ['Angry','Fear','Happy','Sad','Surprise']

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Tabs
tab1, tab2 = st.tabs(["📂 Upload Image", "📷 Camera"])

def preprocess(face):
    face = cv2.resize(face, (128,128))
    face = face / 255.0
    face = face.reshape(1,128,128,1)
    return face

def predict_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        st.error("❌ Face not detected")
        return

    x, y, w, h = faces[0]
    face = gray[y:y+h, x:x+w]

    # Center small face image
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        st.image(face, caption="Detected Face", width=150)

    processed = preprocess(face)
    preds = model.predict(processed)
    idx = np.argmax(preds)
    emotion = emotion_labels[idx]
    emoji_dict = {
        'Angry': '😠',
        'Fear': '😨',
        'Happy': '😊',
        'Sad': '😢',
        'Surprise': '😲'
    }
    emoji = emoji_dict.get(emotion, '')

    st.success(f" {emoji} Emotion: {emotion_labels[idx]}")
    st.write(f"Confidence: {float(preds[0][idx])*100:.2f}%")

# ------------------- TAB 1 -------------------
with tab1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Center small uploaded image
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption="Uploaded Image", width=250)

        if st.button("Predict Emotion"):
            predict_image(img)

# ------------------- TAB 2 -------------------
with tab2:
    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:
        file_bytes = np.frombuffer(camera_image.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.image(img, caption="Captured Image", width=250)

        if st.button("Predict from Camera"):
            predict_image(img)

# Footer
st.markdown("<p style='text-align: center;'>Built with ❤️ by Sahil Katve</p>", unsafe_allow_html=True)

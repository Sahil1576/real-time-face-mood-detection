import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# ------------------------------
# Load Model
# ------------------------------
@st.cache_resource
def load_my_model():
    return load_model("Now_the_best_model.keras")

model = load_my_model()

# Your emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Suprise']

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# ------------------------------
# Page Config
# ------------------------------
st.set_page_config(page_title="Mood Detection", layout="centered")

st.markdown(
    "<h1 style='text-align:center; color:#FF4B4B;'>😊 Mood Detection App</h1>",
    unsafe_allow_html=True
)

# ------------------------------
# Face Detection + Preprocessing
# ------------------------------
def detect_face(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    # Select largest face
    faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
    (x, y, w, h) = faces[0]

    face = gray[y:y+h, x:x+w]
    return face


def preprocess(face):
    face = cv2.resize(face, (128, 128))
    face = face / 255.0
    face = face.reshape(1, 128, 128, 1)
    return face


def predict(img):
    face = detect_face(img)

    if face is None:
        return None, None, None

    processed = preprocess(face)

    preds = model.predict(processed)
    idx = np.argmax(preds)

    label = emotion_labels[idx]
    confidence = float(preds[0][idx])

    return label, confidence, face


# ------------------------------
# Tabs
# ------------------------------
tab1, tab2 = st.tabs(["📁 Upload Image", "📷 Camera"])

# ------------------------------
# Upload Section
# ------------------------------
with tab1:
    file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

    if file:
        image = Image.open(file)
        img = np.array(image)

        # Show small image
        st.image(image, width=250)

        if st.button("Predict Emotion"):
            label, conf, face = predict(img)

            if face is None:
                st.error("❌ No face detected")
            else:
                st.image(face, caption="Detected Face", width=200)

                st.success(f"😊 Emotion: {label}")
                st.write(f"Confidence: {conf:.2%}")

# ------------------------------
# Camera Section
# ------------------------------
with tab2:
    cam = st.camera_input("Take a picture")

    if cam:
        image = Image.open(cam)
        img = np.array(image)

        st.image(image, width=250)

        if st.button("Predict from Camera"):
            label, conf, face = predict(img)

            if face is None:
                st.error("❌ No face detected")
            else:
                st.image(face, caption="Detected Face", width=200)

                st.success(f"😊 Emotion: {label}")
                st.write(f"Confidence: {conf:.2%}")

# ------------------------------
# Footer
# ------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Built with ❤️ by Sahil Katve</p>",
    unsafe_allow_html=True
)
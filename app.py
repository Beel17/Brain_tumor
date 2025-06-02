import streamlit as st
import sqlite3
from PIL import Image
import numpy as np
import tensorflow as tf
import datetime
import os

# Page config
st.set_page_config(
    page_title="Brain Tumor Detection 🧠",
    layout="centered",
    page_icon="🧠"
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/mobilenetv2_brain_tumor.h5")

model = load_model()

# Database
conn = sqlite3.connect("database/predictions.db", check_same_thread=False)
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS predictions (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              filename TEXT,
              result TEXT,
              confidence REAL,
              timestamp DATETIME)''')
conn.commit()

# Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #333446;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 10px;
        padding: 0.5em 2em;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Header
st.title("🧠 Brain Tumor Detection")
st.subheader("Upload an MRI scan for analysis")

# File Upload
uploaded_file = st.file_uploader("Choose an MRI image (PNG, JPG, JPEG)", type=["jpg", "jpeg", "png"])

# Preprocessing
def preprocess_image(image):
    img = Image.open(image).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

if uploaded_file is not None:
    st.image(uploaded_file, caption="📷 Uploaded Image", width=300)

    if st.button("🔍 Analyze"):
        with st.spinner("Analyzing... Please wait ⏳"):
            processed_img = preprocess_image(uploaded_file)
            prediction = model.predict(processed_img)
            confidence = float(prediction[0][0])
            result = "🧠 Tumor Detected" if confidence > 0.5 else "✅ No Tumor"

            c.execute('''INSERT INTO predictions (filename, result, confidence, timestamp)
                         VALUES (?, ?, ?, ?)''',
                      (uploaded_file.name, result, confidence, datetime.datetime.now()))
            conn.commit()

            st.success(f"**Result:** {result}")
            st.metric(label="Confidence Level", value=f"{abs(confidence - 0.5) * 200:.1f}%")

# Sidebar Navigation
st.sidebar.title("🔍 Navigation")
st.sidebar.page_link("app.py", label="Home")
st.sidebar.page_link("pages/history.py", label="Prediction History")

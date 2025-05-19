import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model(
    r"G:\My Drive\Projects\resume Project\MoodScope-CNN\happy_sad_app\models\happy_sad_model.h5"
)

# Prediction function
def predict_image(img):
    img = img.convert('RGB')
    img = img.resize((200, 200))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "😊 Happy" if prediction[0][0] < 0.5 else "😔 Sad"

# ----- STYLING -----
st.set_page_config(page_title="Happy or Sad Classifier", layout="wide")
st.markdown("""
    <style>
        body {
            background: linear-gradient(135deg, #f0f4f8, #dfe6e9);
        }
        .main-title {
            text-align: center;
            font-size: 3.5em;
            font-weight: bold;
            color: #2c3e50;
            margin-top: 20px;
        }
        .subtitle {
            text-align: center;
            color: #636e72;
            font-size: 1.3em;
            margin-bottom: 40px;
        }
        .card {
            background-color: #ffffff;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
        }
        .result {
            text-align: center;
            font-size: 2em;
            margin-top: 20px;
            color: #0984e3;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            font-size: 0.9em;
            color: #b2bec3;
            margin-top: 50px;
        }
    </style>
""", unsafe_allow_html=True)

# ----- SIDEBAR -----
with st.sidebar:
    st.title("ℹ️ About App")
    st.write("""
    This app uses a deep learning model to classify images of faces as either **Happy** or **Sad**.

    ✅ Built with TensorFlow  
    ✅ Deployed using Streamlit  
    ✅ Simple, fast & fun!
    """)
    st.markdown("---")
    st.write("👨‍💻 Developed by: [Premkumar Pawar]")

# ----- MAIN CONTENT -----
st.markdown("<div class='main-title'>🎭 Happy or Sad Classifier</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an image of a face and see the emotion prediction instantly</div>", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="📸 Preview", use_column_width=True)

        with st.spinner("Analyzing..."):
            result = predict_image(img)

        st.markdown(f"<div class='result'>Result: {result}</div>", unsafe_allow_html=True)
    else:
        st.info("⬆️ Upload a .jpg or .png image to get started")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>© 2025 Happy-Sad Classifier | Powered by TensorFlow & Streamlit</div>", unsafe_allow_html=True)

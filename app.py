import os
import json
import numpy as np
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
import tensorflow as tf
from streamlit_drawable_canvas import st_canvas

# ---------------- CONFIG ----------------
st.set_page_config(page_title="MathLens", layout="wide")

st.markdown("""
<style>
.stApp { background: #f5f9ff; }

.main-header {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    padding: 2rem;
    border-radius: 15px;
    color: white;
    text-align: center;
}

.custom-card {
    background: white;
    padding: 1.5rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
}

.prediction-card {
    background: linear-gradient(135deg, #667eea, #764ba2);
    padding: 2rem;
    border-radius: 15px;
    text-align: center;
    color: white;
}

.prediction-symbol {
    font-size: 5rem;
    font-weight: bold;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 2px solid #e6ecf5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("models/digit_math_recognizer.h5")
    with open("models/label_map.json") as f:
        label_map = json.load(f)
        label_map = {int(k): v for k, v in label_map.items()}
    return model, label_map

model, label_map = load_model()

# ---------------- PREPROCESS ----------------
def preprocess(img):
    if img.mode != 'L':
        img = img.convert('L')

    img = img.resize((28, 28))
    img = ImageEnhance.Contrast(img).enhance(2.5)
    img = img.filter(ImageFilter.GaussianBlur(1))

    arr = np.array(img)

    if np.mean(arr) > 127:
        arr = 255 - arr

    arr = arr / 255.0
    return arr.reshape(1,28,28,1)

# ---------------- HEADER ----------------
st.markdown("""
<div class="main-header">
<h1>🧠 MathLens</h1>
<p>AI Handwritten Digit & Math Recognizer</p>
</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🧠 MathLens Guide")
    st.markdown("---")

    # HOW TO USE
    st.markdown("### ✏️ How to Use")
    st.write("1. Draw a digit or symbol")
    st.write("2. Click Predict")
    st.write("3. View result & confidence")
    st.write("4. Check probability bars")

    st.markdown("---")

    # SUPPORTED SYMBOLS
    st.markdown("### 🔢 Supported Symbols")
    st.write("Digits: 0 – 9")
    st.write("Operators:")
    st.write("➕ Plus (+)")
    st.write("➖ Minus (-)")
    st.write("✖️ Multiply (x)")
    st.write("➗ Divide (/)")
    st.write("🟰 Equals (=)")

    st.markdown("---")

    # MODEL INFO (🔥 NEW)
    st.markdown("### 🤖 Model Details")

    st.write("**Architecture:**")
    st.write("- Convolutional Neural Network (CNN)")
    st.write("- 3 Conv Blocks + BatchNorm + Dropout")

    st.write("**Dataset:**")
    st.write("- MNIST (digits 0–9)")
    st.write("- Synthetic symbols (+, -, x, /, =)")

    st.write("**Classes:**")
    st.write("- Total: 15")
    st.write("- 10 digits + 5 operators")

    st.write("**Input Format:**")
    st.write("- 28 × 28 grayscale image")

    st.write("**Training:**")
    st.write("- Data Augmentation used")
    st.write("- EarlyStopping + LR Scheduler")

    st.write("**Performance:**")
    st.write("- Accuracy: ~97%–99%")
    st.write("- Optimizer: Adam")
    st.write("- Loss: Categorical Crossentropy")

    st.markdown("---")

    # TECH STACK
    st.markdown("### ⚙️ Tech Stack")
    st.write("- TensorFlow / Keras")
    st.write("- OpenCV")
    st.write("- Streamlit")
    st.write("- NumPy")

    st.markdown("---")

    # TIPS
    st.markdown("### 💡 Tips for Better Accuracy")
    st.write("- Draw thick strokes")
    st.write("- Center the symbol")
    st.write("- Avoid very small drawings")

# ---------------- STATE ----------------
if "prediction" not in st.session_state:
    st.session_state.prediction = None
    st.session_state.confidence = None
    st.session_state.probs = None

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([1,1])

# ---------------- LEFT SIDE ----------------
with col1:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("✏️ Draw Digit")

    canvas = st_canvas(
        stroke_width=20,
        stroke_color="white",
        background_color="black",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    predict_btn = st.button("🔮 Predict")
    clear_btn = st.button("🧹 Clear")

    if clear_btn:
        st.session_state.prediction = None
        st.session_state.probs = None
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)

    # Upload
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("📤 Upload Image")

    file = st.file_uploader("Upload", type=["png","jpg","jpeg"])

    if file:
        img = Image.open(file)
        st.image(img, width=150)

        if st.button("Predict Uploaded"):
            img = preprocess(img)
            pred = model.predict(img)[0]

            st.session_state.probs = pred
            idx = np.argmax(pred)

            st.session_state.prediction = label_map[idx]
            st.session_state.confidence = float(np.max(pred)*100)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- RIGHT SIDE ----------------
with col2:
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.subheader("🎯 Prediction")

    if predict_btn and canvas.image_data is not None:
        img = canvas.image_data.astype(np.uint8)

        if len(img.shape) == 3:
            img = img[:,:,0]

        img = Image.fromarray(img)
        img = preprocess(img)

        pred = model.predict(img)[0]

        st.session_state.probs = pred
        idx = np.argmax(pred)

        st.session_state.prediction = label_map[idx]
        st.session_state.confidence = float(np.max(pred)*100)

    # Show prediction
    if st.session_state.prediction:
        st.markdown(f"""
        <div class="prediction-card">
            <div class="prediction-symbol">{st.session_state.prediction}</div>
            <p>Confidence: {st.session_state.confidence:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)

        # ✅ FIXED PROBABILITY BARS
    # ✅ FIXED PROBABILITY BARS
    st.subheader("📊 Confidence Distribution")

    probs = st.session_state.probs

    if probs is not None:   # 🔥 THIS FIXES YOUR ERROR
        for i in range(len(probs)):
            st.progress(float(probs[i]), text=f"{label_map[i]} ({probs[i]*100:.1f}%)")
    else:
         st.info("Draw and click Predict to see confidence")

    st.markdown('</div>', unsafe_allow_html=True)
# ---------------- FOOTER ----------------
st.markdown("""
<hr>
<center>🚀 Built with TensorFlow & Streamlit</center>
""", unsafe_allow_html=True)
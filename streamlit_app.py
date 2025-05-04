import streamlit as st
import numpy as np
import joblib
import tensorflow as tf

# Load model
scaler = joblib.load("scaler.pkl")
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# --- Sidebar Info ---
st.sidebar.title("📘 Navigasi")
st.sidebar.markdown("""
- 📱 Prediksi Harga
- 🧠 ANN Model
- 📊 Data Kaggle
""")

# --- Header ---
st.title("📱 Aplikasi Prediksi Harga Smartphone")
st.markdown("Gunakan aplikasi ini untuk memprediksi *kelas harga* smartphone berdasarkan spesifikasi teknisnya.")

st.image("brand.png", use_container_width=True)

with st.expander("ℹ️ Tentang Aplikasi"):
    st.write("""
    Aplikasi ini menggunakan model Artificial Neural Network (ANN) untuk memprediksi kisaran harga smartphone.
    Dataset diambil dari Kaggle: [Mobile Price Classification](https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification).
    """)

# --- Input Layout ---
st.subheader("🔧 Input Spesifikasi Smartphone")

col1, col2 = st.columns(2)

with col1:
    battery_power = st.slider("Battery Power (mAh)", 500, 2000, 1000)
    blue = st.selectbox("Bluetooth", [0, 1])
    clock_speed = st.slider("Clock Speed (GHz)", 0.5, 3.0, 1.5)
    dual_sim = st.selectbox("Dual SIM", [0, 1])
    fc = st.slider("Front Camera (MP)", 0, 20, 5)
    four_g = st.selectbox("4G", [0, 1])
    int_memory = st.slider("Internal Memory (GB)", 2, 64, 16)
    m_dep = st.slider("Mobile Depth (cm)", 0.1, 1.0, 0.5)
    mobile_wt = st.slider("Weight (grams)", 80, 200, 120)
    n_cores = st.slider("CPU Cores", 1, 8, 4)

with col2:
    ram = st.slider("RAM (MB)", 256, 4000, 2000)
    pc = st.slider("Primary Camera (MP)", 0, 20, 10)
    px_height = st.slider("Pixel Height", 0, 2000, 800)
    px_width = st.slider("Pixel Width", 0, 2000, 1200)
    sc_h = st.slider("Screen Height (cm)", 5, 20, 12)
    sc_w = st.slider("Screen Width (cm)", 0, 20, 8)
    talk_time = st.slider("Talk Time (hours)", 2, 20, 10)
    three_g = st.selectbox("3G", [0, 1])
    touch_screen = st.selectbox("Touch Screen", [0, 1])
    wifi = st.selectbox("WiFi", [0, 1])

# --- Prediksi ---
if st.button("🔍 Prediksi Harga"):
    input_array = np.array([[battery_power, blue, clock_speed, dual_sim, fc, four_g, int_memory,
                             m_dep, mobile_wt, n_cores, ram, pc, px_height, px_width, sc_h, sc_w,
                             talk_time, three_g, touch_screen, wifi]])
    
    input_scaled = scaler.transform(input_array)

    interpreter.set_tensor(input_details[0]['index'], input_scaled.astype(np.float32))
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    pred = np.argmax(output)

    kelas = {
        0: "💸 Kelas 0: Murah",
        1: "💵 Kelas 1: Menengah Bawah",
        2: "💰 Kelas 2: Menengah Atas",
        3: "💎 Kelas 3: Mahal"
    }

    st.success(f"📊 Prediksi kelas harga: {kelas[pred]}")

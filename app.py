import streamlit as st
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid
import json
import gdown

st.title("🍅 Klasifikasi Penyakit Daun Tomat")

uploaded_file = st.file_uploader("Pilih gambar daun tomat (JPG/PNG)", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Simpan sementara agar bisa diproses Keras
    file_path = os.path.join("temp_upload", f"{uuid.uuid4().hex}.png")
    os.makedirs("temp_upload", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Tampilkan gambar
    st.image(file_path, caption="Gambar yang diupload", use_column_width=True)

    # Preprocessing gambar
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    pred = model.predict(img_array)[0]
    top_index = np.argmax(pred)
    result = labels[top_index] if top_index < len(labels) else "Tidak diketahui"

    # Probabilitas
    prediction_list = [(labels[i], f"{p*100:.2f}%") for i, p in enumerate(pred)]
    description = deskripsi_penyakit.get(result, "Deskripsi tidak tersedia.")

    # Tampilkan hasil
    st.subheader("Hasil Prediksi:")
    st.write(f"**{result}**")
    st.write(description)

    st.subheader("Probabilitas setiap kelas:")
    for label, prob in prediction_list:
        st.write(f"{label}: {prob}")

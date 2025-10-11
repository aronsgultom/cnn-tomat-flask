from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import uuid
import json
import gdown

# =========================================
# Unduh model dari Google Drive jika belum ada
# =========================================
MODEL_PATH = "model_tomat.keras"
DRIVE_ID = "1ZhMlGHwufTj3TKwhrXaoEA89N0HEYrHj"  # 🔁 ganti dengan ID model kamu di Google Drive

if not os.path.exists(MODEL_PATH):
    print("📥 Mengunduh model dari Google Drive...")
    url = f"https://drive.google.com/uc?id={DRIVE_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)
    print("✅ Model berhasil diunduh.")

# =========================================
# Inisialisasi Flask
# =========================================
app = Flask(__name__)

# =========================================
# Load model dan label map
# =========================================
print("🔄 Memuat model dan label map...")
model = load_model(MODEL_PATH)

LABEL_MAP_PATH = 'label_map.json'
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"File label map tidak ditemukan: {LABEL_MAP_PATH}")
with open(LABEL_MAP_PATH) as f:
    label_map = json.load(f)

# Urutkan label berdasarkan indeks
labels = [None] * len(label_map)
for label, index in label_map.items():
    labels[index] = label.replace('_', ' ').capitalize()

# =========================================
# Deskripsi penyakit
# =========================================
deskripsi_penyakit = {
    'Early blight': "Early blight ditandai dengan bercak gelap konsentris dan menyebar dari daun bawah. Disebabkan oleh jamur Alternaria solani.",
    'Late blight': "Late blight menyebabkan bercak gelap dan basah. Penyakit ini menyebar cepat dan menyebabkan daun menguning dan membusuk.",
    'Leaf mold': "Leaf mold muncul sebagai bercak kuning di atas daun dan jamur abu-abu di bawah daun. Umumnya terjadi di tempat lembap.",
    'Healthy': "Daun dalam kondisi sehat. Tidak ditemukan gejala penyakit."
}

# =========================================
# Konfigurasi upload folder
# =========================================
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# =========================================
# ROUTE: Halaman utama
# =========================================
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return "Tidak ada file yang diupload.", 400

        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in ['.jpg', '.jpeg', '.png']:
            return "Format file tidak didukung. Gunakan JPG/PNG.", 400

        # Simpan file upload
        filename = f"{uuid.uuid4().hex}{ext}"
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            # Preprocessing gambar
            img = image.load_img(file_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Prediksi model
            pred = model.predict(img_array)[0]
            top_index = np.argmax(pred)
            result = labels[top_index] if top_index < len(labels) else "Tidak diketahui"

            # Buat daftar probabilitas
            prediction_list = [(labels[i], f"{p * 100:.2f}%") for i, p in enumerate(pred)]
            description = deskripsi_penyakit.get(result, "Deskripsi tidak tersedia.")

            return render_template(
                'result.html',
                label=result,
                image_path='/' + file_path.replace("\\", "/"),
                predictions=prediction_list,
                description=description
            )

        except Exception as e:
            return f"Terjadi kesalahan saat memproses gambar: {str(e)}", 500

    return render_template('upload.html')

# =========================================
# MAIN ENTRY POINT
# =========================================
if __name__ == '__main__':
    # Jangan gunakan debug=True di Render
    app.run(host='0.0.0.0', port=10000)


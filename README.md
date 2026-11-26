# ErgoVision AI — Posture & Eye-Strain Monitor

Aplikasi web berbasis **Computer Vision** (`YOLO11n-pose`) dan **FastAPI** untuk memantau ergonomi postur duduk serta jarak pandang pengguna terhadap layar secara *real-time* melalui webcam browser — **100% in-memory (Zero-Storage Privacy)**.

Dirancang dengan estetika **Winter Mesh Gradient** (`#192641`, `#596779`, `#A1BDD0`, `#D2E0EE`, `#F8F9FA`) yang bersih, tenang, dan mudah dipindai.

---

## Fitur Utama

- **Analisis Postur Real-Time (`yolo11n-pose.pt`):**
  - **Kemiringan Bahu (*Shoulder Tilt*):** Menghitung sudut bahu kiri dan kanan terhadap sumbu horizontal (ambang ideal $\le 6.5^\circ$).
  - **Deteksi *Tech Neck* (*Neck Ratio*):** Rasio tinggi vertikal hidung ke titik tengah bahu terhadap lebar bahu (peringatan bila turun $> 18\%$ dari acuan).
  - **Deteksi Jarak Layar (*Eye-Strain Proximity*):** Rasio jarak antar-mata (atau lebar bahu sebagai fallback) terhadap lebar frame (peringatan bila meningkat $> 25\%$ dari acuan).
- **Skor Ergonomi Dinamis ($0 - 100$):** Penilaian proporsional dengan status *Postur Ergonomis* ($\ge 80$), *Perlu Penyesuaian* ($55 - 79$), dan *Postur Berisiko* ($< 55$).
- **Kalibrasi Postur Ideal 1-Klik:** Menyimpan acuan personal (`neck_ratio` & `face_ratio`) untuk sesi aktif.
- **Statistik Sesi Duduk (*Live Timer*):** Mencatat akumulasi durasi (`MM:SS`) postur sehat vs membungkuk.
- **Notifikasi Suara Lembut (*Ambient Chime*):** Sintesis nada Web Audio API tanpa file eksternal yang berbunyi satu kali jika postur buruk bertahan $\pm 3$ detik, dilengkapi tombol Mute/Unmute.
- **Zero-Storage Privacy & Keamanan Input:** Semua gambar diproses murni di RAM (`np.frombuffer` + `cv2.imdecode`), validasi MIME (`JPEG`/`PNG`/`WEBP`), dan batas maksimal upload `5 MB`.

---

## Struktur Direktori

```text
ErgoVision-website/
├── templates/
│   └── index.html        # Antarmuka web responsif (Winter Mesh Gradient) & logika kamera/audio
├── main.py               # Server FastAPI, validasi keamanan, & mesin analisis geometri YOLO
├── requirements.txt      # Daftar dependensi Python dengan versi terkunci
├── Dockerfile            # Konfigurasi container & pre-download model weights
└── README.md             # Dokumentasi instalasi & deployment
```

---

## Menjalankan Secara Lokal

### Prasyarat
- Python `3.11+`
- Webcam (desktop atau mobile browser)

### Langkah Instalasi

```bash
# 1. (Opsional) Buat dan aktifkan virtual environment
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate

# 2. Instal dependensi
pip install -r requirements.txt

# 3. Jalankan server FastAPI di port 7860
uvicorn main:app --host 0.0.0.0 --port 7860 --reload
```

Buka browser dan kunjungi: **`http://localhost:7860`**

---

## Menjalankan dengan Docker

```bash
# Build image (otomatis mengunduh model yolo11n-pose.pt saat build)
docker build -t ergovision-ai .

# Jalankan container pada port 7860
docker run -p 7860:7860 ergovision-ai
```

---

## Panduan Deployment (24 Jam Self-Running)

### 1. Hugging Face Spaces (Docker SDK)
1. Buat Space baru di Hugging Face dengan SDK **Docker (Blank)**.
2. Tambahkan header YAML berikut di bagian paling atas `README.md` Space Anda:
   ```yaml
   ---
   title: ErgoVision AI
   emoji: 👁️
   colorFrom: indigo
   colorTo: blue
   sdk: docker
   app_port: 7860
   ---
   ```
3. Push `Dockerfile`, `requirements.txt`, `main.py`, dan `templates/index.html` ke repository Space.

### 2. Railway
1. Hubungkan repository GitHub ke proyek baru di Railway.
2. Railway otomatis mendeteksi `Dockerfile`.
3. Set variabel lingkungan `PORT=7860` (atau biarkan expose port `7860`) dan tambahkan Healthcheck Path ke `/health`.

---

## Spesifikasi Endpoint API

| Method | Path | Deskripsi |
| :--- | :--- | :--- |
| `GET` | `/` | Merender halaman antarmuka utama `templates/index.html` |
| `GET` | `/health` | Memeriksa status server dan kesiapan model YOLO |
| `POST` | `/api/analyze` | Menerima `multipart/form-data` (`file`, opsional `baseline_neck_ratio`, `baseline_face_ratio`) dan mengembalikan skor, metrik ergonomi, isu, serta gambar skeleton teranotasi (base64 JPEG) |

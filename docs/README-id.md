# Cek Kandungan Gula Makanan — Prototipe Aplikasi

Prototipe akademik (Streamlit) untuk proyek *Food Classification Based on Nutritions to
Recognize Diabetes Risks*.

| Bagian | Fungsi | Cara kerja | Dikerjakan |
|---|---|---|---|
| Foto → nama makanan | Mengenali 101 jenis makanan | Deep learning (EfficientNetB0, transfer learning) | Anggota 1 |
| Nama makanan → nutrisi | Gula, karbohidrat, serat, energi per 100 g | Tabel referensi | Anggota 2 |
| Nutrisi → penilaian makanan | Rendah / sedang / tinggi gula | Aturan dari batas harian WHO | Anggota 4 |
| Profil kesehatan → risiko diabetes | Risiko rendah / tinggi | Deep learning (MLP) | Anggota 3 |
| Aplikasi & integrasi | Menyatukan semuanya, pengujian | Streamlit | Anggota 4 |

Yang dinilai dari foto adalah **makanannya**, bukan risiko diabetes seseorang. Satu makanan
tidak menentukan apakah seseorang terkena diabetes. Kalau profil kesehatan sudah diisi dan
hasilnya risiko tinggi, penilaian makanan menambahkan peringatan.

> Bukan alat diagnosis medis.

## Aturan penilaian makanan (src/food_risk.py)

Batas WHO yang dipakai:
- **Gula bebas** < 10% energi harian ≈ **50 g** (untuk ~2.000 kkal); idealnya < 5% ≈ **25 g**.
  Sumber: WHO, *Guideline: Sugars intake for adults and children* (2015);
  [WHO fact sheet Healthy diet](https://www.who.int/news-room/fact-sheets/detail/healthy-diet).
- **Serat alami** minimal **25 g** per hari.
  Sumber: [WHO, guideline karbohidrat 2023](https://www.who.int/news/item/17-07-2023-who-updates-guidelines-on-fats-and-carbohydrates).
- **Karbohidrat**: WHO tidak menetapkan batas maksimal; yang dianjurkan adalah sumbernya
  (biji-bijian utuh, sayur, buah, kacang-kacangan). Karena itu karbohidrat ditampilkan tanpa
  kategori, dan diberi pengingat bila gulanya rendah.

Kategori gula per 100 g (aturan kelompok, diturunkan dari batas ideal WHO 25 g):

| Gula per 100 g | Kategori | Alasan |
|---|---|---|
| ≥ 12,5 g | Tinggi gula | ≥ separuh batas ideal harian |
| 5 – 12,5 g | Gula sedang | 20–50% batas ideal harian |
| < 5 g | Rendah gula | < 20% batas ideal harian |

Pada tabel saat ini: 19 makanan tinggi gula (semuanya kue/dessert), 10 sedang, 72 rendah.

Nutrisi yang ditampilkan dipilih yang paling berpengaruh untuk gula darah: **gula,
karbohidrat, serat**, ditambah **energi** sebagai konteks. Natrium tidak ditampilkan (lihat
catatan data di bawah).

## Struktur folder

```
food-diabetes-app/
├── app.py                      # tampilan Streamlit (tab Foto makanan + Profil kesehatan)
├── .streamlit/config.toml      # tema warna
├── requirements.txt
├── src/
│   ├── food_service.py         # memuat model makanan, prediksi top-3
│   ├── nutrition_service.py    # mencari nutrisi di tabel
│   ├── food_risk.py            # aturan WHO untuk menilai makanan
│   ├── diabetes_service.py     # memuat model diabetes + validasi input
│   └── keras_compat.py         # memuat model dari Keras versi lebih baru (fix error Colab)
├── models/
│   ├── diabetes/               # dari Anggota 3
│   └── food/                   # dari Anggota 1 (food_model.keras di-upload terpisah)
├── data/food_nutrition.csv     # dari Anggota 2 (per 100 g)
└── tests/test_services.py      # 25 tes otomatis
```

## Cara menjalankan

Di Colab: pakai notebook `Jalankan_App_di_Colab.ipynb`.

Di laptop (Python 3.11/3.12):
```bash
pip install -r requirements.txt
python -m pytest -q        # harus 25 passed
streamlit run app.py
```

## Hasil uji dengan foto asli

Diuji dengan 4 foto makanan asli (bukan dari dataset Food-101):

| Foto | Prediksi teratas | Benar? |
|---|---|---|
| Hamburger | Hamburger 92% | Ya |
| Sushi | Sushi 100% | Ya |
| Pizza (difoto bersama orang) | Pizza 51%, Paella 46% | Ya, tapi ragu → aplikasi meminta pengguna memilih |
| Steak | Prime Rib 87%, Steak 5% | Tidak (kelas yang mirip) |

## Functional testing

Otomatis (`pytest`, 25 tes): validasi input diabetes, peringatan data di luar sebaran,
tabel nutrisi (file hilang, kolom salah, nilai kosong), kecocokan tabel dengan 101 kelas
model, kontrak model makanan, kompatibilitas Keras, dan kategori gula WHO
(termasuk batas tepat 5 g dan 12,5 g).

Manual:

| No | Skenario | Hasil yang diharapkan |
|----|----------|-----------------------|
| 1 | Upload foto makanan yang didukung | Nama makanan, label nutrisi, kategori gula |
| 2 | Upload foto dessert (mis. cheesecake) | Kategori "Tinggi gula" |
| 3 | Upload file rusak / bukan gambar | Pesan error, aplikasi tidak crash |
| 4 | Foto bukan makanan / foto ragu | Keyakinan < 70% → pengguna memilih dari 3 kemungkinan |
| 5 | Isi profil kesehatan nilai wajar | Risiko rendah/tinggi + persentase |
| 6 | Profil risiko tinggi, lalu upload dessert | Penilaian makanan menambahkan peringatan |
| 7 | Isi gula darah sangat rendah/tinggi | Catatan bahwa nilai di luar sebaran data training |

## Catatan data dan keterbatasan (untuk laporan)
- **Angka nutrisi dari Anggota 2 masih berupa perkiraan**, belum per item dari USDA (kolom
  `source_id` semuanya `REF-100G`). Sebelum laporan final, sebaiknya diverifikasi.
- **Natrium di tabel tidak dipakai**: nilainya dihitung dengan rumus `kalori × 2` di
  `build.py` Anggota 2, bukan data asli.
- Batas WHO berlaku untuk **gula bebas**, sedangkan tabel memuat **gula total** (termasuk
  gula alami buah/susu). Penilaian jadi sedikit lebih ketat dari seharusnya.
- Nutrisi adalah perkiraan per 100 g dari tabel, bukan pengukuran dari foto.
- Model diabetes dilatih pada dataset Kaggle yang labelnya tidak selalu konsisten dengan
  kriteria klinis; hasil tidak valid untuk diagnosis.
- Urutan kelas `cheesecake` / `cheese_plate` perlu dikonfirmasi ke Anggota 1.

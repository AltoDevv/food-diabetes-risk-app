"""
Food classification service — KONTRAK untuk Anggota 1.

Yang harus diserahkan Anggota 1 ke folder models/food/:
  1. food_model.keras   -> model Keras yang sudah dilatih
  2. class_names.json   -> list nama kelas, URUTANNYA sama dengan output model
                           contoh: ["fried_rice", "hamburger", "pizza", ...]

Aturan preprocessing (WAJIB disepakati):
  - Input model: gambar RGB 224x224, nilai piksel 0–255 (float32).
  - Normalisasi (misal mobilenet_v2.preprocess_input / Rescaling) DIMASUKKAN
    ke dalam model sebagai layer, supaya aplikasi tidak perlu tahu backbone apa
    yang dipakai. Contoh di Colab:

        inputs = tf.keras.Input((224, 224, 3))
        x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
        x = base_model(x, training=False)
        ...

Kalau file belum ada, aplikasi tetap jalan dan menampilkan pesan "belum tersedia".
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "food" / "food_model.keras"
DEFAULT_CLASSES = ROOT / "models" / "food" / "class_names.json"
IMG_SIZE = (224, 224)


def is_available(model_path: Path = DEFAULT_MODEL, classes_path: Path = DEFAULT_CLASSES) -> bool:
    return Path(model_path).exists() and Path(classes_path).exists()


def preprocess(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(img, dtype=np.float32)  # 0–255, normalisasi ada di dalam model
    return np.expand_dims(arr, 0)


class FoodService:
    def __init__(self, model_path: Path = DEFAULT_MODEL, classes_path: Path = DEFAULT_CLASSES):
        from src.keras_compat import load_model_compat

        # load_model_compat: aman walau model disimpan dengan Keras yang lebih baru
        self.model = load_model_compat(model_path)
        with open(classes_path, encoding="utf-8") as f:
            self.class_names: list[str] = json.load(f)

        n_out = self.model.output_shape[-1]
        if n_out != len(self.class_names):
            raise ValueError(
                f"Jumlah output model ({n_out}) != jumlah class_names ({len(self.class_names)})."
            )

    def predict(self, image: Image.Image, top_k: int = 3) -> list[dict]:
        probs = self.model.predict(preprocess(image), verbose=0)[0]
        # Kalau layer terakhir belum softmax, ubah logits jadi probabilitas.
        if probs.min() < 0 or not np.isclose(probs.sum(), 1.0, atol=1e-3):
            e = np.exp(probs - probs.max())
            probs = e / e.sum()
        idx = np.argsort(probs)[::-1][:top_k]
        return [{"label": self.class_names[i], "confidence": float(probs[i])} for i in idx]

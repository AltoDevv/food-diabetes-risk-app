"""
Diabetes Risk service (wrapper untuk model dari Anggota 3).

Perbedaan dengan predict_diabetes.py asli:
- Kategori yang tidak dikenal TIDAK diam-diam diganti 0, tapi melempar ValueError.
- Ada validasi rentang angka dasar supaya input aneh (misal BMI 500) ditolak.
- Menyediakan daftar pilihan kategori langsung dari encoder, supaya form di
  aplikasi selalu cocok dengan data training.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = ROOT / "models" / "diabetes" / "diabetes_mlp_model.keras"
DEFAULT_PIPELINE = ROOT / "models" / "diabetes" / "diabetes_pipeline.pkl"

# Rentang input yang masuk akal (validasi kewajaran, BUKAN batas medis).
NUMERIC_RANGES = {
    "age": (18, 100),
    "bmi": (10.0, 70.0),
    "hours_sleep_per_night": (0.0, 24.0),
    "stress_level": (1, 10),
    "fasting_blood_sugar": (40, 500),
    "hba1c_level": (3.0, 20.0),
    "blood_pressure_systolic": (60, 260),
    "blood_pressure_diastolic": (30, 160),
    "waist_circumference_cm": (40.0, 200.0),
}

THRESHOLD = 0.5


class DiabetesService:
    def __init__(self, model_path: Path = DEFAULT_MODEL, pipeline_path: Path = DEFAULT_PIPELINE):
        from src.keras_compat import load_model_compat

        self.model = load_model_compat(model_path)
        pipeline = joblib.load(pipeline_path)
        self.scaler = pipeline["scaler"]
        self.encoders = pipeline["encoders"]
        self.feature_names: list[str] = list(pipeline["feature_names"])

    # ---------- info untuk membangun form ----------
    def categorical_options(self) -> dict[str, list[str]]:
        return {col: [str(c) for c in le.classes_] for col, le in self.encoders.items()}

    def numeric_features(self) -> list[str]:
        return [f for f in self.feature_names if f not in self.encoders]

    # ---------- validasi ----------
    def validate(self, data: dict) -> None:
        missing = [f for f in self.feature_names if f not in data]
        if missing:
            raise ValueError(f"Fitur belum diisi: {missing}")

        for col, le in self.encoders.items():
            val = str(data[col]).strip()
            if val not in le.classes_:
                raise ValueError(
                    f"Nilai '{val}' untuk '{col}' tidak dikenal. Pilihan: {list(le.classes_)}"
                )

        for col in self.numeric_features():
            try:
                v = float(data[col])
            except (TypeError, ValueError):
                raise ValueError(f"'{col}' harus berupa angka, didapat: {data[col]!r}")
            if col in NUMERIC_RANGES:
                lo, hi = NUMERIC_RANGES[col]
                if not lo <= v <= hi:
                    raise ValueError(f"'{col}' = {v} di luar rentang wajar ({lo}–{hi}).")

    def out_of_distribution(self, data: dict, z_limit: float = 3.0) -> list[str]:
        """Fitur numerik yang jauh dari sebaran data training (|z| > z_limit).
        Prediksi untuk input seperti ini kurang bisa dipercaya."""
        idx = {f: i for i, f in enumerate(self.feature_names)}
        flagged = []
        for f in self.numeric_features():
            i = idx[f]
            z = (float(data[f]) - self.scaler.mean_[i]) / self.scaler.scale_[i]
            if abs(z) > z_limit:
                flagged.append(f)
        return flagged

    # ---------- prediksi ----------
    def predict(self, data: dict) -> dict:
        self.validate(data)

        row = {}
        for f in self.feature_names:
            if f in self.encoders:
                row[f] = int(self.encoders[f].transform([str(data[f]).strip()])[0])
            else:
                row[f] = float(data[f])
        df = pd.DataFrame([row], columns=self.feature_names)

        x = self.scaler.transform(df)
        prob = float(self.model.predict(x, verbose=0)[0][0])
        high = prob >= THRESHOLD
        return {
            "risk_probability": round(prob, 4),
            "risk_percentage": round(prob * 100, 2),
            "prediction": "High Risk" if high else "Low Risk",
            "threshold": THRESHOLD,
            "out_of_distribution": self.out_of_distribution(data),
        }

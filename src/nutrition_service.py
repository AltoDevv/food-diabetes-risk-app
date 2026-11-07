"""
Nutritional mapping service — membaca tabel dari Anggota 2.

File: data/food_nutrition.csv  (format dari Anggota 2)
Kolom wajib:
  food_label, display_name, serving_size_g, calories_kcal, carbohydrates_g, sugar_g,
  fiber_g, protein_g, fat_g, saturated_fat_g, sodium_mg, source_name, source_id, source_url

- food_label harus sama persis dengan isi models/food/class_names.json.
- Sel nutrisi KOSONG berarti "belum diisi", BUKAN 0.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TABLE = ROOT / "data" / "food_nutrition.csv"

NUTRIENT_COLUMNS = [
    "serving_size_g", "calories_kcal", "carbohydrates_g", "sugar_g", "fiber_g",
    "protein_g", "fat_g", "saturated_fat_g", "sodium_mg",
]
REQUIRED_COLUMNS = (
    ["food_label", "display_name"] + NUTRIENT_COLUMNS + ["source_name", "source_id", "source_url"]
)

DISPLAY_NAMES = {
    "serving_size_g": "Porsi referensi (g)",
    "calories_kcal": "Kalori (kkal)",
    "carbohydrates_g": "Karbohidrat (g)",
    "sugar_g": "Gula (g)",
    "fiber_g": "Serat (g)",
    "protein_g": "Protein (g)",
    "fat_g": "Lemak (g)",
    "saturated_fat_g": "Lemak jenuh (g)",
    "sodium_mg": "Natrium (mg)",
}


def normalize_label(label: str) -> str:
    return str(label).strip().lower().replace(" ", "_").replace("-", "_")


def _is_blank(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


class NutritionService:
    def __init__(self, table_path: Path = DEFAULT_TABLE):
        self.table_path = Path(table_path)
        if not self.table_path.exists():
            self.df = pd.DataFrame(columns=REQUIRED_COLUMNS).set_index("food_label")
            return
        df = pd.read_csv(self.table_path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"food_nutrition.csv kurang kolom: {missing}")
        df["food_label"] = df["food_label"].map(normalize_label)
        if df["food_label"].duplicated().any():
            dup = df.loc[df["food_label"].duplicated(), "food_label"].tolist()
            raise ValueError(f"food_label duplikat: {dup}")
        for c in NUTRIENT_COLUMNS:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        self.df = df.set_index("food_label")

    def is_empty(self) -> bool:
        return len(self.df) == 0

    def display_name(self, food_label: str) -> str:
        key = normalize_label(food_label)
        if key in self.df.index and not _is_blank(self.df.loc[key, "display_name"]):
            return str(self.df.loc[key, "display_name"])
        return key.replace("_", " ").title()

    def lookup(self, food_label: str) -> dict:
        """
        status:
          "not_found" -> label tidak ada di tabel
          "empty"     -> label ada, tapi semua nilai nutrisi masih kosong
          "ok"        -> ada nilai nutrisi (sebagian boleh kosong)
        """
        key = normalize_label(food_label)
        if key not in self.df.index:
            return {"status": "not_found", "food_label": key}
        row = self.df.loc[key].to_dict()
        values = {c: (None if _is_blank(row[c]) else float(row[c])) for c in NUTRIENT_COLUMNS}
        status = "empty" if all(v is None for v in values.values()) else "ok"
        source = " · ".join(
            str(row[c]) for c in ("source_name", "source_id", "source_url") if not _is_blank(row[c])
        )
        return {
            "status": status,
            "food_label": key,
            "display_name": self.display_name(key),
            "values": values,
            "source": source,
        }

    def missing_classes(self, class_names: list[str]) -> list[str]:
        """Kelas yang bisa diprediksi model tapi belum punya nilai nutrisi."""
        return [c for c in class_names if self.lookup(c)["status"] != "ok"]

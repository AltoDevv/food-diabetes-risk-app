"""
Penilaian makanan berdasarkan batas harian WHO (aturan, BUKAN model AI).

Yang dinilai adalah MAKANANNYA (per 100 g porsi referensi), bukan risiko diabetes
seseorang. Satu makanan tidak menentukan apakah seseorang terkena diabetes.

Sumber WHO:
- Gula bebas < 10% energi harian ≈ 50 g untuk ~2.000 kkal; idealnya < 5% ≈ 25 g.
  WHO Guideline: Sugars intake for adults and children (2015) dan fact sheet
  "Healthy diet" — https://www.who.int/news-room/fact-sheets/detail/healthy-diet
- Serat alami minimal 25 g per hari (WHO, Carbohydrate intake guideline 2023) —
  https://www.who.int/news/item/17-07-2023-who-updates-guidelines-on-fats-and-carbohydrates
- WHO TIDAK menetapkan batas maksimal karbohidrat total; yang ditekankan adalah
  kualitasnya (biji-bijian utuh, sayur, buah, kacang-kacangan).

Aturan kelompok (diturunkan dari angka WHO di atas, per 100 g):
- Gula >= 12,5 g  -> "tinggi"  (>= separuh batas ideal harian 25 g)
- Gula 5–12,5 g   -> "sedang"  (20–50% batas ideal harian)
- Gula < 5 g      -> "rendah"  (< 20% batas ideal harian)

Catatan: angka gula di tabel adalah GULA TOTAL. Batas WHO berlaku untuk GULA BEBAS
(gula tambahan, madu, sirup, jus). Untuk makanan berbahan buah/susu, perbandingan ini
cenderung lebih ketat dari seharusnya (konservatif).
"""
from __future__ import annotations

WHO_SUGAR_MAX_G = 50.0       # < 10% energi, ~2.000 kkal
WHO_SUGAR_IDEAL_G = 25.0     # < 5% energi
WHO_FIBER_MIN_G = 25.0       # serat alami per hari
REFERENCE_KCAL = 2000.0      # acuan energi harian yang dipakai WHO

SUGAR_HIGH_G = WHO_SUGAR_IDEAL_G * 0.5   # 12,5 g
SUGAR_MEDIUM_G = WHO_SUGAR_IDEAL_G * 0.2  # 5 g

LEVELS = {
    "tinggi": "Tinggi gula",
    "sedang": "Gula sedang",
    "rendah": "Rendah gula",
}


def idn(x: float) -> str:
    """Format angka gaya Indonesia: 14.8 -> '14,8'."""
    return f"{x:g}".replace(".", ",")


def _pct(value: float | None, ref: float) -> float | None:
    return None if value is None else round(value / ref * 100, 1)


def sugar_level(sugar_g: float) -> str:
    if sugar_g >= SUGAR_HIGH_G:
        return "tinggi"
    if sugar_g >= SUGAR_MEDIUM_G:
        return "sedang"
    return "rendah"


def assess(values: dict, personal_high_risk: bool | None = None) -> dict:
    """
    values: dict nutrisi per 100 g (boleh ada yang None), kunci yang dipakai:
            sugar_g, carbohydrates_g, fiber_g, calories_kcal
    personal_high_risk: hasil formulir kesehatan (True/False) atau None kalau belum diisi.
    """
    sugar = values.get("sugar_g")
    carbs = values.get("carbohydrates_g")
    fiber = values.get("fiber_g")
    kcal = values.get("calories_kcal")

    out = {
        "sugar_g": sugar,
        "carbs_g": carbs,
        "fiber_g": fiber,
        "kcal": kcal,
        "sugar_pct_max": _pct(sugar, WHO_SUGAR_MAX_G),
        "sugar_pct_ideal": _pct(sugar, WHO_SUGAR_IDEAL_G),
        "fiber_pct": _pct(fiber, WHO_FIBER_MIN_G),
        "kcal_pct": _pct(kcal, REFERENCE_KCAL),
        "level": None,
        "title": "Data gula belum tersedia",
        "message": "Angka gula untuk makanan ini belum diisi di tabel nutrisi, "
                   "jadi kandungan gulanya belum bisa dinilai.",
    }
    if sugar is None:
        return out

    level = sugar_level(sugar)
    out["level"] = level
    out["title"] = LEVELS[level]

    if level == "tinggi":
        msg = (f"100 g makanan ini mengandung {idn(sugar)} g gula, lebih dari separuh batas "
               f"gula harian yang dianjurkan WHO ({idn(WHO_SUGAR_IDEAL_G)} g). "
               "Sebaiknya dibatasi, terutama bagi yang berisiko atau memiliki diabetes.")
    elif level == "sedang":
        msg = (f"100 g makanan ini mengandung {idn(sugar)} g gula, sekitar "
               f"{idn(out['sugar_pct_ideal'])}% dari batas gula harian yang dianjurkan WHO "
               f"({idn(WHO_SUGAR_IDEAL_G)} g). Konsumsi secukupnya.")
    else:
        msg = f"Kandungan gulanya rendah ({idn(sugar)} g per 100 g)."
        if carbs is not None and carbs > 0:
            msg += (f" Tetap perhatikan karbohidratnya ({idn(carbs)} g per 100 g), "
                    "karena karbohidrat juga menaikkan gula darah.")

    if personal_high_risk is True and level in ("tinggi", "sedang"):
        msg += (" Formulir kesehatan Anda menunjukkan risiko diabetes tinggi, "
                "jadi makanan seperti ini perlu lebih diperhatikan.")

    out["message"] = msg
    return out

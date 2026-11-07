"""
Prototipe: cek kandungan gula makanan dari foto + risiko diabetes dari profil kesehatan.
Jalankan:  streamlit run app.py
"""
import html
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

from src import food_risk, food_service
from src.diabetes_service import NUMERIC_RANGES, DiabetesService
from src.nutrition_service import NutritionService

st.set_page_config(page_title="Cek gula makanan", layout="centered")

LOW_CONFIDENCE = 0.70  # sama dengan threshold Anggota 2

LEVEL_COLOR = {"tinggi": "#A23B2A", "sedang": "#B7862B", "rendah": "#3E7C59", None: "#6B7770"}

# ---------------------------------------------------------------- gaya
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..900&display=swap');

.stApp, .stApp p, .stApp label, .stApp input, .stApp button, .stApp textarea,
.stApp h1, .stApp h2, .stApp h3, .stApp li, .stApp [data-baseweb="tab"],
.app-head, .nlabel, .verdict, .meta, .footnote {
  font-family: 'Archivo', system-ui, sans-serif;
}
.block-container { padding-top: 2.5rem; max-width: 820px; }
header[data-testid="stHeader"] { background: transparent; }

.app-head h1 {
  font-stretch: 75%; font-weight: 800; font-size: 2.6rem; line-height: 1.05;
  letter-spacing: -0.01em; margin: 0 0 .5rem 0; color: #1E2A24;
}
.app-head p { font-size: 1.02rem; color: #46524B; max-width: 60ch; margin: 0 0 1.25rem 0; }

[data-baseweb="tab-list"] { gap: 1.5rem; }
[data-baseweb="tab"] { font-size: 1rem; padding-left: 0; padding-right: 0; }

/* ---- label nutrisi ---- */
.nlabel {
  background: #FFFFFF; border: 2px solid #1E2A24; padding: 14px 16px 12px;
  color: #1E2A24;
}
.nlabel .name {
  font-stretch: 70%; font-weight: 900; font-size: 2rem; line-height: 1; margin: 0;
}
.nlabel .sub { font-size: .88rem; color: #46524B; margin: 6px 0 8px; }
.nlabel .thick { border-top: 8px solid #1E2A24; margin: 6px 0 4px; }
.nlabel .row { border-top: 1px solid #1E2A24; padding: 7px 0 6px; }
.nlabel .row:first-of-type { border-top: none; }
.nlabel .line { display: flex; justify-content: space-between; align-items: baseline; }
.nlabel .k { font-weight: 800; font-size: 1.02rem; }
.nlabel .v { font-weight: 800; font-size: 1.02rem; font-variant-numeric: tabular-nums; }
.nlabel .note { font-size: .8rem; color: #46524B; margin-top: 3px; }
.wbar { position: relative; height: 8px; background: #E8ECE6; margin-top: 6px; }
.wfill { position: absolute; left: 0; top: 0; bottom: 0; }
.wtick { position: absolute; top: -3px; bottom: -3px; width: 2px; background: #1E2A24; }
.nlabel .small .k, .nlabel .small .v { font-weight: 500; font-size: .92rem; }

.verdict { border-left: 6px solid; background: #FFFFFF; padding: 12px 16px; margin-top: 14px; }
.verdict .t { font-weight: 800; font-size: 1.15rem; margin: 0 0 4px; }
.verdict p { margin: 0; font-size: .95rem; line-height: 1.5; color: #1E2A24; }

.meta { font-size: .9rem; line-height: 1.45; color: #46524B; margin: 10px 0 0; }
.footnote { font-size: .8rem; color: #6B7770; margin-top: 3rem; padding-top: 1rem;
  border-top: 1px solid #D6DBD5; max-width: 70ch; line-height: 1.5; }
</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------- loaders
@st.cache_resource(show_spinner="Memuat model risiko diabetes…")
def load_diabetes():
    return DiabetesService()


@st.cache_resource(show_spinner="Memuat model pengenal makanan…")
def load_food():
    return food_service.FoodService() if food_service.is_available() else None


@st.cache_resource
def load_nutrition():
    return NutritionService()


def esc(x) -> str:
    return html.escape(str(x))


def fmt(v, unit: str) -> str:
    return "–" if v is None else f"{food_risk.idn(v)} {unit}"


idn = food_risk.idn

# Pilihan formulir: nilai asli dataset (Inggris) -> teks tampilan (Indonesia), urut logis.
CHOICES = {
    "gender": {"Female": "Perempuan", "Male": "Laki-laki", "Other": "Lainnya"},
    "family_history_diabetes": {"No": "Tidak", "Yes": "Ya"},
    "physical_activity_level": {"Sedentary": "Jarang bergerak", "Moderate": "Sedang", "Active": "Aktif"},
    "diet_type": {"Non-Vegetarian": "Makan daging", "Vegetarian": "Vegetarian",
                  "Vegan": "Vegan", "Pescatarian": "Pescatarian (ikan, tanpa daging)"},
    "smoking_status": {"Never": "Tidak pernah", "Former": "Sudah berhenti", "Current": "Masih merokok"},
    "alcohol_consumption": {"Never": "Tidak pernah", "Occasional": "Kadang-kadang", "Regular": "Rutin"},
    "income_bracket": {"Low": "Rendah", "Middle": "Menengah", "High": "Tinggi"},
}


def choice(label: str, field: str, opts: dict):
    allowed = opts[field]
    names = CHOICES.get(field, {})
    ordered = [v for v in names if v in allowed] + [v for v in allowed if v not in names]
    return st.selectbox(label, ordered, format_func=lambda v: names.get(v, v))


def bar(value, scale, color, tick=None) -> str:
    if value is None:
        return ""
    w = max(0.0, min(100.0, value / scale * 100))
    t = f'<div class="wtick" style="left:{tick / scale * 100:.1f}%"></div>' if tick else ""
    return f'<div class="wbar"><div class="wfill" style="width:{w:.1f}%;background:{color}"></div>{t}</div>'


def nutrition_label(name: str, a: dict, confidence: float | None) -> str:
    color = LEVEL_COLOR[a["level"]]
    sub = "Perkiraan kandungan per 100 g"
    if confidence is not None:
        sub += f", dikenali dengan keyakinan {confidence:.0%}"

    sugar_note = ("Belum ada data." if a["sugar_g"] is None else
                  f"{idn(a['sugar_pct_max'])}% dari batas harian WHO ({idn(food_risk.WHO_SUGAR_MAX_G)} g). "
                  f"Garis menandai batas ideal {idn(food_risk.WHO_SUGAR_IDEAL_G)} g.")
    fiber_note = ("Belum ada data." if a["fiber_g"] is None else
                  f"{idn(a['fiber_pct'])}% dari anjuran harian WHO ({idn(food_risk.WHO_FIBER_MIN_G)} g). "
                  "Serat membantu memperlambat naiknya gula darah.")
    kcal_note = "" if a["kcal"] is None else f"{idn(a['kcal_pct'])}% dari 2.000 kkal sehari"

    return f"""
<div class="nlabel">
  <p class="name">{esc(name)}</p>
  <p class="sub">{esc(sub)}</p>
  <div class="thick"></div>
  <div class="row">
    <div class="line"><span class="k">Gula</span><span class="v">{fmt(a['sugar_g'], 'g')}</span></div>
    {bar(a['sugar_g'], food_risk.WHO_SUGAR_MAX_G, color, tick=food_risk.WHO_SUGAR_IDEAL_G)}
    <div class="note">{esc(sugar_note)}</div>
  </div>
  <div class="row">
    <div class="line"><span class="k">Karbohidrat</span><span class="v">{fmt(a['carbs_g'], 'g')}</span></div>
    <div class="note">WHO tidak menetapkan batas maksimal karbohidrat, tetapi menganjurkan
    sumber dari biji-bijian utuh, sayur, buah, dan kacang-kacangan.</div>
  </div>
  <div class="row">
    <div class="line"><span class="k">Serat</span><span class="v">{fmt(a['fiber_g'], 'g')}</span></div>
    {bar(a['fiber_g'], food_risk.WHO_FIBER_MIN_G, '#2F5D50')}
    <div class="note">{esc(fiber_note)}</div>
  </div>
  <div class="row small">
    <div class="line"><span class="k">Energi</span><span class="v">{fmt(a['kcal'], 'kkal')}</span></div>
    <div class="note">{esc(kcal_note)}</div>
  </div>
</div>
<div class="verdict" style="border-color:{color}">
  <p class="t" style="color:{color}">{esc(a['title'])}</p>
  <p>{esc(a['message'])}</p>
</div>
"""


# ---------------------------------------------------------------- header
st.markdown(
    """
<div class="app-head">
  <h1>Cek kandungan gula makanan</h1>
  <p>Unggah foto makanan untuk melihat perkiraan gula, karbohidrat, dan serat per 100 g,
  dibandingkan dengan batas harian dari WHO. Isi profil kesehatan untuk melihat perkiraan
  risiko diabetes Anda.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_food, tab_health = st.tabs(["Foto makanan", "Profil kesehatan"])
personal = st.session_state.get("personal_high_risk")  # None / True / False

# ================================================================ TAB 1: FOOD
with tab_food:
    food, nutrition = None, None
    try:
        food = load_food()
    except Exception as e:
        st.error(f"Model pengenal makanan tidak bisa dimuat: {e}")
    try:
        nutrition = load_nutrition()
    except Exception as e:
        st.error(f"Tabel nutrisi tidak valid: {e}")

    if food is None:
        st.info("Model pengenal makanan belum dipasang. Letakkan food_model.keras dan "
                "class_names.json di folder models/food.")

    uploaded = st.file_uploader("Foto makanan (JPG atau PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is not None and food is not None:
        try:
            image = Image.open(uploaded)
            image.load()
            image = ImageOps.exif_transpose(image)  # foto HP sering tersimpan miring
        except (UnidentifiedImageError, OSError):
            st.error("File ini bukan gambar yang bisa dibaca. Pilih file JPG atau PNG lain.")
            image = None

        if image is not None:
            with st.spinner("Mengenali makanan…"):
                preds = food.predict(image, top_k=3)
            top = preds[0]

            def pretty(label):
                return nutrition.display_name(label) if nutrition else label.replace("_", " ").title()

            col_img, col_label = st.columns([5, 6], gap="large")
            with col_img:
                st.image(image, use_container_width=True)
                chosen = top["label"]
                if top["confidence"] < LOW_CONFIDENCE:
                    st.warning("Foto ini kurang jelas dikenali. Pilih makanan yang paling sesuai.")
                    chosen = st.selectbox(
                        "Makanan di foto", [p["label"] for p in preds],
                        format_func=lambda l: f"{pretty(l)} ({next(p['confidence'] for p in preds if p['label']==l):.0%})",
                    )
                else:
                    others = ", ".join(f"{pretty(p['label'])} ({p['confidence']:.0%})" for p in preds[1:])
                    st.markdown(f'<div class="meta">Kemungkinan lain: {esc(others)}</div>',
                                unsafe_allow_html=True)

            with col_label:
                info = nutrition.lookup(chosen) if nutrition else {"status": "not_found"}
                if info["status"] == "ok":
                    a = food_risk.assess(info["values"], personal_high_risk=personal)
                else:
                    a = food_risk.assess({"sugar_g": None})
                conf = top["confidence"] if chosen == top["label"] else None
                st.markdown(nutrition_label(pretty(chosen), a, conf), unsafe_allow_html=True)
                if personal is None:
                    st.markdown('<div class="meta">Isi tab Profil kesehatan agar penilaian ini '
                                'ikut mempertimbangkan kondisi Anda.</div>', unsafe_allow_html=True)

# ================================================================ TAB 2: HEALTH
with tab_health:
    try:
        diab = load_diabetes()
    except Exception as e:
        diab = None
        st.error(f"Model risiko diabetes tidak bisa dimuat: {e}")

    if diab is not None:
        opts = diab.categorical_options()
        st.markdown('<div class="meta">Isi sesuai hasil pemeriksaan terakhir Anda. '
                    'Semua kolom wajib diisi.</div>', unsafe_allow_html=True)

        with st.form("health_form", border=False):
            c1, c2 = st.columns(2, gap="large")
            with c1:
                age = st.number_input("Usia (tahun)", *NUMERIC_RANGES["age"], value=45, step=1)
                gender = choice("Jenis kelamin", "gender", opts)
                bmi = st.number_input("BMI", *NUMERIC_RANGES["bmi"], value=23.0, step=0.1, format="%.1f")
                waist = st.number_input("Lingkar pinggang (cm)",
                                        *NUMERIC_RANGES["waist_circumference_cm"], value=90.0, step=0.5, format="%.1f")
                family = choice("Ada keluarga dengan diabetes?", "family_history_diabetes", opts)
                activity = choice("Aktivitas fisik", "physical_activity_level", opts)
                diet = choice("Pola makan", "diet_type", opts)
                income = choice("Kelompok pendapatan", "income_bracket", opts)
            with c2:
                smoking = choice("Merokok", "smoking_status", opts)
                alcohol = choice("Konsumsi alkohol", "alcohol_consumption", opts)
                sleep = st.number_input("Jam tidur per malam",
                                        *NUMERIC_RANGES["hours_sleep_per_night"], value=7.0, step=0.5, format="%.1f")
                stress = st.slider("Tingkat stres (1–10)", *NUMERIC_RANGES["stress_level"], value=5)
                fbs = st.number_input("Gula darah puasa (mg/dL)",
                                      *NUMERIC_RANGES["fasting_blood_sugar"], value=100, step=1)
                hba1c = st.number_input("HbA1c (%)", *NUMERIC_RANGES["hba1c_level"], value=5.5, step=0.1, format="%.1f")
                sys_bp = st.number_input("Tekanan darah sistolik (mmHg)",
                                         *NUMERIC_RANGES["blood_pressure_systolic"], value=120, step=1)
                dia_bp = st.number_input("Tekanan darah diastolik (mmHg)",
                                         *NUMERIC_RANGES["blood_pressure_diastolic"], value=80, step=1)
            submitted = st.form_submit_button("Hitung risiko", type="primary")

        if submitted:
            data = {
                "age": age, "gender": gender, "bmi": bmi,
                "family_history_diabetes": family, "physical_activity_level": activity,
                "diet_type": diet, "smoking_status": smoking, "alcohol_consumption": alcohol,
                "hours_sleep_per_night": sleep, "stress_level": stress,
                "fasting_blood_sugar": fbs, "hba1c_level": hba1c,
                "blood_pressure_systolic": sys_bp, "blood_pressure_diastolic": dia_bp,
                "waist_circumference_cm": waist, "income_bracket": income,
            }
            try:
                st.session_state["health_result"] = diab.predict(data)
                st.session_state["personal_high_risk"] = (
                    st.session_state["health_result"]["prediction"] == "High Risk")
            except ValueError as e:
                st.error(str(e))

        res = st.session_state.get("health_result")
        if res:
            high = res["prediction"] == "High Risk"
            color = LEVEL_COLOR["tinggi"] if high else LEVEL_COLOR["rendah"]
            title = "Risiko diabetes tinggi" if high else "Risiko diabetes rendah"
            body = (f"Model memperkirakan peluang {res['risk_percentage']:.0f}% "
                    f"(batas kategori {res['threshold'] * 100:.0f}%). ")
            body += ("Disarankan memeriksakan diri ke tenaga kesehatan dan membatasi makanan "
                     "tinggi gula." if high else
                     "Tetap jaga pola makan dan aktivitas fisik.")
            if res["out_of_distribution"]:
                body += (" Catatan: beberapa nilai yang Anda isi jauh dari data yang dipakai "
                         "melatih model, jadi perkiraan ini kurang bisa diandalkan.")
            st.markdown(
                f'<div class="verdict" style="border-color:{color}">'
                f'<p class="t" style="color:{color}">{esc(title)}</p><p>{esc(body)}</p>'
                f'{bar(res["risk_probability"], 1.0, color, tick=res["threshold"])}</div>',
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="footnote">Prototipe tugas kuliah, bukan alat diagnosis. Nilai nutrisi adalah '
    'perkiraan per 100 g dari tabel referensi, bukan hasil pengukuran dari foto. Batas gula '
    'WHO berlaku untuk gula bebas (gula tambahan, madu, sirup, jus), sedangkan tabel memuat '
    'gula total, sehingga penilaian ini cenderung lebih ketat.</div>',
    unsafe_allow_html=True,
)

"""
Functional test. Jalankan dari root project:  python -m pytest -q
"""
import json

import numpy as np
import pytest
from PIL import Image

from src import food_service
from src.diabetes_service import DiabetesService
from src.nutrition_service import REQUIRED_COLUMNS, NutritionService

VALID = {
    "age": 45, "gender": "Male", "bmi": 27.0, "family_history_diabetes": "Yes",
    "physical_activity_level": "Sedentary", "diet_type": "Non-Vegetarian",
    "smoking_status": "Never", "alcohol_consumption": "Never",
    "hours_sleep_per_night": 7.0, "stress_level": 5, "fasting_blood_sugar": 150,
    "hba1c_level": 6.8, "blood_pressure_systolic": 135, "blood_pressure_diastolic": 85,
    "waist_circumference_cm": 100.0, "income_bracket": "Middle",
}


# ------------------------------------------------------------ diabetes
@pytest.fixture(scope="module")
def diab():
    return DiabetesService()


def test_diabetes_valid_input(diab):
    r = diab.predict(VALID)
    assert 0.0 <= r["risk_probability"] <= 1.0
    assert r["prediction"] in {"High Risk", "Low Risk"}


def test_diabetes_has_16_features(diab):
    assert len(diab.feature_names) == 16


def test_diabetes_unknown_category_rejected(diab):
    with pytest.raises(ValueError, match="tidak dikenal"):
        diab.predict({**VALID, "gender": "Laki-laki"})


def test_diabetes_missing_field_rejected(diab):
    bad = dict(VALID)
    bad.pop("bmi")
    with pytest.raises(ValueError, match="belum diisi"):
        diab.predict(bad)


def test_diabetes_out_of_range_rejected(diab):
    with pytest.raises(ValueError, match="rentang"):
        diab.predict({**VALID, "bmi": 500})


def test_diabetes_non_numeric_rejected(diab):
    with pytest.raises(ValueError, match="angka"):
        diab.predict({**VALID, "age": "empat puluh"})


# ------------------------------------------------------------ nutrition
def _write_table(path, rows):
    import pandas as pd
    pd.DataFrame(rows, columns=REQUIRED_COLUMNS).to_csv(path, index=False)


def test_nutrition_missing_file_is_empty(tmp_path):
    n = NutritionService(tmp_path / "tidak_ada.csv")
    assert n.is_empty() and n.lookup("pizza")["status"] == "not_found"


def test_nutrition_lookup_statuses(tmp_path):
    # Angka di sini DUMMY hanya untuk tes, bukan data nutrisi asli.
    p = tmp_path / "t.csv"
    blank = [None] * 9
    _write_table(p, [
        ["pizza", "Pizza", 100, 1, 1, 1, 1, 1, 1, None, 1, "DUMMY", "D-1", "https://example.com"],
        ["sushi", "Sushi", *blank, None, None, None],
    ])
    n = NutritionService(p)
    ok = n.lookup("Pizza")  # dinormalisasi -> "pizza"
    assert ok["status"] == "ok"
    assert ok["values"]["calories_kcal"] == 1.0
    assert ok["values"]["saturated_fat_g"] is None  # kosong tetap kosong, bukan 0
    assert "DUMMY" in ok["source"]
    assert n.lookup("sushi")["status"] == "empty"
    assert n.lookup("ramen")["status"] == "not_found"
    assert n.missing_classes(["pizza", "sushi", "ramen"]) == ["sushi", "ramen"]


def test_real_nutrition_table_matches_model_classes():
    """Tabel Anggota 2 harus mencakup semua kelas model Anggota 1, dengan urutan class_index sama."""
    import json
    import pandas as pd
    from src.nutrition_service import DEFAULT_TABLE
    from src.food_service import DEFAULT_CLASSES

    if not (DEFAULT_TABLE.exists() and DEFAULT_CLASSES.exists()):
        pytest.skip("file asli belum ada")
    classes = json.loads(DEFAULT_CLASSES.read_text())
    df = pd.read_csv(DEFAULT_TABLE).sort_values("class_index")
    assert df["food_label"].tolist() == classes


def test_nutrition_bad_columns_rejected(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("food_name,calories\npizza,1\n")
    with pytest.raises(ValueError, match="kurang kolom"):
        NutritionService(p)


# ------------------------------------------------------------ food (pakai model dummy)
@pytest.fixture
def dummy_food_model(tmp_path):
    import tensorflow as tf

    classes = ["fried_rice", "pizza", "sushi"]
    inp = tf.keras.Input((224, 224, 3))
    x = tf.keras.layers.Rescaling(1 / 255.0)(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    out = tf.keras.layers.Dense(len(classes), activation="softmax")(x)
    m = tf.keras.Model(inp, out)
    mp, cp = tmp_path / "food_model.keras", tmp_path / "class_names.json"
    m.save(mp)
    cp.write_text(json.dumps(classes))
    return mp, cp, classes


def test_food_contract(dummy_food_model):
    mp, cp, classes = dummy_food_model
    assert food_service.is_available(mp, cp)
    svc = food_service.FoodService(mp, cp)
    img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
    preds = svc.predict(img, top_k=3)
    assert len(preds) == 3
    assert preds[0]["label"] in classes
    assert abs(sum(p["confidence"] for p in preds) - 1.0) < 1e-3


def test_food_class_count_mismatch_rejected(dummy_food_model):
    mp, cp, _ = dummy_food_model
    cp.write_text(json.dumps(["a", "b"]))
    with pytest.raises(ValueError, match="class_names"):
        food_service.FoodService(mp, cp)


def test_food_preprocess_handles_png_with_alpha():
    img = Image.new("RGBA", (50, 80))
    assert food_service.preprocess(img).shape == (1, 224, 224, 3)


def test_diabetes_out_of_distribution_flag(diab):
    r = diab.predict({**VALID, "fasting_blood_sugar": 45})
    assert "fasting_blood_sugar" in r["out_of_distribution"]
    assert diab.predict(VALID)["out_of_distribution"] == []


def test_keras_compat_strips_unknown_config_keys(tmp_path):
    """Simulasi model dari Keras versi lebih baru: config punya parameter yang tidak dikenal."""
    import json
    import zipfile
    import tensorflow as tf
    from src.keras_compat import load_model_compat

    inp = tf.keras.Input((4,))
    out = tf.keras.layers.Dense(2, activation="softmax")(inp)
    src = tmp_path / "m.keras"
    tf.keras.Model(inp, out).save(src)

    bad = tmp_path / "future.keras"
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(bad, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == "config.json":
                cfg = json.loads(data)
                for layer in cfg["config"]["layers"]:
                    if layer["class_name"] == "Dense":
                        layer["config"]["fitur_dari_masa_depan"] = None
                data = json.dumps(cfg)
            zout.writestr(item, data)

    m = load_model_compat(bad)
    assert m.output_shape == (None, 2)


# ------------------------------------------------------------ food risk (aturan WHO)
from src import food_risk


@pytest.mark.parametrize("sugar,level", [
    (0, "rendah"), (4.9, "rendah"), (5, "sedang"), (12.4, "sedang"), (12.5, "tinggi"), (40, "tinggi"),
])
def test_food_risk_levels(sugar, level):
    assert food_risk.sugar_level(sugar) == level


def test_food_risk_percentages_use_who_limits():
    a = food_risk.assess({"sugar_g": 25, "fiber_g": 5, "carbohydrates_g": 50, "calories_kcal": 400})
    assert a["sugar_pct_max"] == 50.0      # 25 / 50 g
    assert a["sugar_pct_ideal"] == 100.0   # 25 / 25 g
    assert a["fiber_pct"] == 20.0          # 5 / 25 g
    assert a["kcal_pct"] == 20.0           # 400 / 2000 kkal


def test_food_risk_low_sugar_mentions_carbs():
    a = food_risk.assess({"sugar_g": 1, "carbohydrates_g": 22})
    assert a["level"] == "rendah" and "karbohidrat" in a["message"]


def test_food_risk_missing_sugar():
    a = food_risk.assess({"sugar_g": None})
    assert a["level"] is None


def test_food_risk_personal_high_risk_note():
    a = food_risk.assess({"sugar_g": 20}, personal_high_risk=True)
    assert "risiko diabetes tinggi" in a["message"]
    b = food_risk.assess({"sugar_g": 20}, personal_high_risk=False)
    assert "risiko diabetes tinggi" not in b["message"]

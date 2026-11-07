"""
Memuat file .keras yang disimpan dengan Keras versi LEBIH BARU di Keras versi lebih lama
(misalnya model dilatih dengan Keras 3.16-dev, tapi Colab memakai Keras yang lebih lama).

Penyebab error: versi baru menambahkan parameter di config (contoh: VarianceScaling
`input_axes`, `output_axes`) yang tidak dikenal versi lama, sehingga muncul
"__init__() got an unexpected keyword argument 'input_axes'".

Solusi: buang parameter yang tidak dikenal dari config.json lalu muat ulang.
Bobot (model.weights.h5) TIDAK diubah sama sekali.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import zipfile
from pathlib import Path

_UNEXPECTED_KW = re.compile(r"unexpected keyword argument '([A-Za-z0-9_]+)'")
# contoh: "Unrecognized keyword arguments passed to BatchNormalization: {'renorm': False, ...}"
_UNRECOGNIZED = re.compile(r"Unrecognized keyword arguments passed to \w+: \{(.*?)\}")


def _bad_keys(err: Exception) -> list[str]:
    msg = str(err)
    m = _UNRECOGNIZED.search(msg)
    if m:
        return re.findall(r"'([A-Za-z0-9_]+)':", m.group(1))
    m = _UNEXPECTED_KW.search(msg)
    return [m.group(1)] if m else []


def _strip_key(obj, key: str) -> int:
    """Hapus `key` dari setiap dict 'config' milik objek Keras ter-serialisasi."""
    removed = 0
    if isinstance(obj, dict):
        cfg = obj.get("config")
        if "class_name" in obj and isinstance(cfg, dict) and key in cfg:
            del cfg[key]
            removed += 1
        for v in obj.values():
            removed += _strip_key(v, key)
    elif isinstance(obj, list):
        for v in obj:
            removed += _strip_key(v, key)
    return removed


def _rewrite(src: Path, dst: Path, config: dict) -> None:
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w", zipfile.ZIP_STORED) as zout:
        for item in zin.infolist():
            if item.filename == "config.json":
                zout.writestr("config.json", json.dumps(config))
            else:
                zout.writestr(item, zin.read(item.filename))


def load_model_compat(path, max_fixes: int = 20, compile: bool = False):
    import keras

    path = Path(path)
    try:
        return keras.models.load_model(path, compile=compile)
    except Exception as first_error:  # noqa: BLE001
        err = first_error

    with zipfile.ZipFile(path) as z:
        config = json.loads(z.read("config.json"))

    removed_keys = []
    tmp_dir = tempfile.mkdtemp()
    tmp_path = Path(tmp_dir) / "patched.keras"
    try:
        for _ in range(max_fixes):
            keys = [k for k in _bad_keys(err) if k not in removed_keys]
            if not keys or sum(_strip_key(config, k) for k in keys) == 0:
                raise err
            removed_keys.extend(keys)
            _rewrite(path, tmp_path, config)
            try:
                model = keras.models.load_model(tmp_path, compile=compile)
                print(f"[keras_compat] Model dimuat setelah membuang parameter: {removed_keys}")
                return model
            except Exception as e:  # noqa: BLE001
                err = e
        raise err
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)
        os.rmdir(tmp_dir)

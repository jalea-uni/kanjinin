#!/usr/bin/env python3
# coding: utf-8
"""
Utils – funzioni comuni per Kanjinin
------------------------------------

Contiene:

* render_template  – disegna un kanji centrato in un canvas numpy
* compute_ssim     – similarità strutturale fra due immagini
* ensure_dir       – crea (ricorsivamente) una directory se non esiste
* load_json / save_json
* set_seed         – fissa il seed di NumPy / random / Torch (se disponibile)

Tutte le funzioni sono pure; non richiedono stato globale.
"""

from __future__ import annotations

import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    # scikit-image for SSIM (migliore qualità)
    from skimage.metrics import structural_similarity as skimage_ssim
except ImportError:  # fallback a None, poi gestito in compute_ssim
    skimage_ssim = None

# --------------------------------------------------------------------------- #
# Rendering di un singolo kanji
# --------------------------------------------------------------------------- #


def render_template(
    char_code: int | str,
    size: Tuple[int, int] = (64, 64),
    font_path: str | None = None,
) -> np.ndarray:
    """
    Crea un'immagine (canvas bianco, scala di grigi) con il kanji centrato.

    Parameters
    ----------
    char_code
        Codice Unicode o carattere vero e proprio da renderizzare.
    size
        Dimensioni del canvas in pixel (w, h).
    font_path
        Path TTF/TTC da usare; se None prova a caricare un font di sistema.

    Returns
    -------
    np.ndarray
        Array uint8 shape (h, w) con valori 0-255.
    """
    # Se viene passato direttamente il carattere lo convertiamo
    if isinstance(char_code, str):
        if len(char_code) != 1:
            raise ValueError("char_code string must be a single character.")
        char = char_code
    else:
        char = chr(char_code)

    # Scegli il font
    default_font_size = int(size[1] * 0.8)
    if font_path:
        try:
            font = ImageFont.truetype(font_path, default_font_size)
        except IOError:
            font = ImageFont.load_default()
    else:
        # Tentativo di font giapponese comuni
        candidate_paths = [
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
            "C:/Windows/Fonts/msgothic.ttc",
        ]
        font = None
        for p in candidate_paths:
            if Path(p).exists():
                try:
                    font = ImageFont.truetype(p, default_font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

    # Canvas bianco
    img_pil = Image.new("L", size, color=255)
    draw = ImageDraw.Draw(img_pil)

    # Centriamo il testo – gestione nuova/vecchia Pillow
    try:
        bbox = draw.textbbox((0, 0), char, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:  # Pillow < 8.0
        w, h = draw.textsize(char, font=font)

    pos = ((size[0] - w) // 2, (size[1] - h) // 2)
    draw.text(pos, char, fill=0, font=font)

    return np.array(img_pil, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Structural SIMilarity fra due immagini
# --------------------------------------------------------------------------- #


def compute_ssim(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Calcola l'SSIM (Structural Similarity) fra due immagini.

    Entrambe vengono convertite in scala di grigi se necessario.

    Notes
    -----
    * Usa scikit-image se disponibile (più accurato);
      altrimenti fallback a OpenCV `cv2.SSIM`.
    """
    if img_a.ndim == 3:
        img_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    if img_b.ndim == 3:
        img_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # Assicuriamoci che abbiano lo stesso shape
    if img_a.shape != img_b.shape:
        raise ValueError("Images must have the same dimensions for SSIM.")

    if skimage_ssim is not None:
        score, _ = skimage_ssim(img_a, img_b, full=True)
        return float(score)

    # Fallback OpenCV 4.10+ (viene esposta da cv2), altrimenti raise
    if hasattr(cv2, "quality") and hasattr(cv2.quality, "QualitySSIM_compute"):
        score = cv2.quality.QualitySSIM_compute(img_a, img_b)[0][0]
        return float(score)

    raise RuntimeError(
        "SSIM calculation requires scikit-image (`pip install scikit-image`) "
        "or OpenCV compiled with the quality module."
    )


# --------------------------------------------------------------------------- #
# Utility varie
# --------------------------------------------------------------------------- #


def ensure_dir(path: str | os.PathLike) -> Path:
    """Crea la directory (*mkdir -p*) e la ritorna come `Path`."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_json(path: str | os.PathLike) -> Any:
    """Carica un file JSON (UTF-8)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str | os.PathLike, *, indent: int = 2) -> None:
    """Salva `obj` in JSON con indentazione (UTF-8)."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def set_seed(seed: int = 42) -> None:
    """Fissa il seed per riproducibilità (random, NumPy e Torch se presente)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------- #
# Debug rapido
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Esegue un piccolo test se lanciato direttamente
    print("Quick self-test utils.py\n")

    kanji_char = "活"
    img = render_template(kanji_char, size=(128, 128))
    tmp_png = Path(__file__).with_name("render_test.png")
    cv2.imwrite(str(tmp_png), img)
    print(f"→ Template salvato in {tmp_png}")

    # SSIM con sé stesso deve dare 1.0
    print(f"SSIM self-compare: {compute_ssim(img, img):.3f}")

    tmp_png.unlink(missing_ok=True)
    print("OK")

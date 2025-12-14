import json
import pathlib
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import cv2
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms


# ---------------------------
# Model definition (from inference/model.py)
# ---------------------------


def get_model(num_classes: int, pretrained: bool = False):
    """ResNet18 1-canale, come nel training originale."""
    model = models.resnet18(pretrained=pretrained)
    # first conv expects 1-ch instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class KanjiEvaluator:
    """
    Carica:
    - best_model.pth (state_dict)
    - label_map.json { "0": "0X0000", "1": "0X3007", ... }

    Ritorna in predict():
    - kanji_unicode (es. '活', '躍', '助', ...)
    - confidence (float)
    """

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = pathlib.Path(model_dir)
        self.device = torch.device("cpu")

        self.model: Optional[torch.nn.Module] = None
        self.label_map: Dict[str, str] = {}
        self.index_to_code: Dict[int, str] = {}
        self.code_to_char: Dict[str, Optional[str]] = {}
        self.model_version: str = "unknown"

        # stessa pipeline del vecchio KanjiEvaluator
        self.img_size = 224
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )

        self._load_model_and_labels()

    # ---------------------------
    # Load model + labels
    # ---------------------------
    def _load_model_and_labels(self):
        model_path = self.model_dir / "best_model.pth"
        label_map_path = self.model_dir / "label_map.json"

        print("[Evaluator] model_path:", model_path, "exists:", model_path.exists())
        print(
            "[Evaluator] label_map_path:",
            label_map_path,
            "exists:",
            label_map_path.exists(),
        )

        # --- load label_map exactly as old code ---
        if not label_map_path.exists():
            print("[Evaluator] label_map.json not found → mock mode")
            self.model = None
            self.model_version = "mock (no label_map)"
            return

        try:
            with label_map_path.open("r", encoding="utf-8") as f:
                self.label_map = json.load(f)
        except Exception as e:
            print("[Evaluator] ERROR loading label_map.json:", e)
            self.model = None
            self.model_version = f"mock (label_map error: {e})"
            return

        # self.label_map: { "0": "0X0000", "1": "0X3007", ... }
        self.index_to_code = {}
        self.code_to_char = {}

        for idx_str, hex_code in self.label_map.items():
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            self.index_to_code[idx] = hex_code

            s = str(hex_code).strip().upper()
            try:
                code_val = int(s, 16)
            except ValueError:
                self.code_to_char[hex_code] = None
                continue

            if code_val == 0:
                # 0X0000 = classe "vuoto"
                self.code_to_char[hex_code] = None
            else:
                self.code_to_char[hex_code] = chr(code_val)

        num_classes = len(self.index_to_code)
        print(f"[Evaluator] Loaded {num_classes} classes from label_map")

        if not model_path.exists():
            print("[Evaluator] best_model.pth not found → mock mode")
            self.model = None
            self.model_version = "mock (no model file)"
            return

        try:
            model = get_model(num_classes=num_classes, pretrained=False)
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()

            self.model = model
            self.model_version = "best_model.pth"
            print("[Evaluator] Model loaded OK (ResNet18 1-ch)")

        except Exception as e:
            print("[Evaluator] ERROR loading model:", e)
            self.model = None
            self.model_version = f"mock (load error: {e})"

    # ---------------------------
    # Preprocess crop -> tensor
    # ---------------------------

# In evaluator.py - aggiungi questa funzione


    def _clean_crop(self, img: np.ndarray) -> np.ndarray:
        """
        Pulisce il crop rimuovendo rumore prima dell'inferenza
        """
        # Assicurati grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Threshold per binarizzare
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Rimuovi piccoli componenti (rumore)
        # Inverti per trovare componenti neri
        inverted = cv2.bitwise_not(binary)

        # Trova componenti connessi
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inverted, connectivity=8
        )

        # Calcola area totale dei componenti (escluso sfondo)
        total_area = sum(stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels))

        # Rimuovi componenti troppo piccoli (< 1% dell'area totale)
        min_area = max(10, total_area * 0.01)

        cleaned = np.zeros_like(binary)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned[labels == i] = 255

        # Inverti di nuovo (kanji nero su bianco)
        cleaned = cv2.bitwise_not(cleaned)

        # 3. Morphological closing per riempire piccoli gap nei tratti
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        return cleaned

    def _crop_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess identical to training:
        - grayscale
        - resize to 64x64
        - ensure kanji is BLACK on WHITE background
        - normalize to [-1,1]
        - shape: B,1,H,W
        """
        img = self._clean_crop(crop)

        # ensure grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # resize to 64x64
        img = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)

        # ✅ VERIFICA AUTOMATICA: il modello vuole kanji NERO su sfondo BIANCO
        # Se l'immagine ha più pixel neri che bianchi → è invertita → correggi
        black_pixels = np.sum(img < 128)
        white_pixels = np.sum(img >= 128)

        if black_pixels > white_pixels:
            # Troppo nero = sfondo nero, kanji bianco → INVERTI
            img = 255 - img

        # normalize to [-1, 1]
        img = img.astype(np.float32) / 255.0  # [0, 1]
        img = (img - 0.5) / 0.5  # [-1, 1]

        img = np.expand_dims(img, axis=0)  # C,H,W
        img = np.expand_dims(img, axis=0)  # B,C,H,W

        return torch.from_numpy(img)

    # ---------------------------
    # Prediction
    # ---------------------------
    def predict(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Ritorna (KANJI, confidence).
        - KANJI è già il carattere Unicode (es. '活', '躍', '助', ...)
        - Se la classe è "vuoto" (0X0000) o non mappata → None
        """
        if self.model is None:
            # mock: densità stroke
            ink = float((crop > 0).sum()) / float(crop.size)
            conf = max(0.1, min(0.99, 1.2 * ink))
            return None, conf

        with torch.no_grad():
            x = self._crop_to_tensor(crop)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        idx = int(pred_idx.item())
        conf = float(confidence.item())

        hex_code = self.index_to_code.get(idx)
        if hex_code is None:
            return None, conf

        char = self.code_to_char.get(hex_code)
        return char, conf

    # ---------------------------
    # Quality metric
    # ---------------------------
    def quality(self, crop_bin: np.ndarray) -> float:
        try:
            blur = cv2.GaussianBlur(crop_bin, (3, 3), 0)
            score, _ = ssim(crop_bin, blur, full=True)
        except Exception:
            ink = float((crop_bin > 0).sum()) / float(crop_bin.size)
            score = ink

        return float(max(0.0, min(1.0, score)))

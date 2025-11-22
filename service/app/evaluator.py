import json
import pathlib
from typing import Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import cv2
from skimage.metrics import structural_similarity as ssim

from torchvision import models


# ---------------------------------------------------------------
#  MODEL ARCHITECTURE — EXACTLY AS YOUR TRAINING
# ---------------------------------------------------------------


def get_model(num_classes: int, pretrained: bool = False):
    """
    Rebuild exactly the trained architecture:
    - ResNet18
    - First conv replaced with 1-channel version
    - fc replaced with num_classes output
    """
    model = models.resnet18(pretrained=pretrained)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------
#  EVALUATOR CLASS
# ---------------------------------------------------------------


class KanjiEvaluator:
    """
    Loads:
    - best_model.pth (state_dict)
    - label_map.json (char ↔ index)
    Provides:
    - predict(crop)
    - quality(crop)
    """

    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = pathlib.Path(model_dir)
        self.model: Optional[torch.nn.Module] = None
        self.labels: Dict[str, Dict[Any, Any]] = {}
        self.model_version: str = "unknown"

        self._load_model_safe()

    # -----------------------------------------------------------
    # Load model + labels
    # -----------------------------------------------------------
    def _load_model_safe(self) -> None:
        model_path = self.model_dir / "best_model.pth"
        label_map_path = self.model_dir / "label_map.json"

        print("[Evaluator] model_path:", model_path, "exists:", model_path.exists())
        print(
            "[Evaluator] label_map_path:",
            label_map_path,
            "exists:",
            label_map_path.exists(),
        )

        # ---------- Load labels ----------
        forward, inverse = {}, {}

        if label_map_path.exists():
            try:
                with label_map_path.open("r", encoding="utf-8") as f:
                    raw = json.load(f)

                # Case A: char -> idx
                if isinstance(raw, dict) and all(
                    isinstance(k, str) and isinstance(v, int) for k, v in raw.items()
                ):
                    forward = dict(raw)
                    inverse = {v: k for k, v in raw.items()}

                # Case B: idx -> char
                elif isinstance(raw, dict):
                    for k, v in raw.items():
                        try:
                            idx = int(k)
                        except ValueError:
                            continue
                        inverse[idx] = v
                        forward[v] = idx

                # Case C: list of chars
                elif isinstance(raw, list):
                    for idx, ch in enumerate(raw):
                        forward[ch] = idx
                        inverse[idx] = ch

                print(f"[Evaluator] Loaded {len(forward)} classes from label_map")

            except Exception as e:
                print("[Evaluator] ERROR loading label_map.json:", e)

        self.labels = {"forward": forward, "inverse": inverse}
        num_classes = len(inverse)

        if num_classes == 0:
            print("[Evaluator] ERROR: No labels found → mock mode")
            self.model = None
            self.model_version = "mock (no labels)"
            return

        # ---------- Load model ----------
        if not model_path.exists():
            print("[Evaluator] best_model.pth not found → mock mode")
            self.model = None
            self.model_version = "mock (missing model)"
            return

        try:
            # 🟢 EXACT architecture
            model = get_model(num_classes=num_classes, pretrained=False)

            state = torch.load(model_path, map_location="cpu")
            model.load_state_dict(state)
            model.eval()

            self.model = model
            self.model_version = "best_model.pth"

            print("[Evaluator] SUCCESS: Loaded trained ResNet18 (1-ch)")

        except Exception as e:
            print("[Evaluator] ERROR loading model:", e)
            self.model = None
            self.model_version = f"mock (load error: {e})"

    # -----------------------------------------------------------
    # Preprocessing  (MATCHING TRAINING PIPELINE)
    # -----------------------------------------------------------
    def _crop_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        """
        Preprocess identical to training:
        - grayscale
        - resize to 224x224
        - normalize to [0,1]
        - shape: B,1,H,W
        """
        img = crop

        # ensure grayscale
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # C,H,W
        img = np.expand_dims(img, axis=0)  # B,C,H,W

        return torch.from_numpy(img)

    # -----------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------
    def predict(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        if self.model is None:
            # mock fallback
            ink = float((crop > 0).sum()) / crop.size
            conf = max(0.1, min(0.99, 1.2 * ink))
            return None, conf

        with torch.no_grad():
            x = self._crop_to_tensor(crop)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        idx = int(np.argmax(probs))
        conf = float(probs[idx])
        char = self.labels["inverse"].get(idx, None)

        return char, conf

    # -----------------------------------------------------------
    # Quality metric
    # -----------------------------------------------------------
    def quality(self, crop_bin: np.ndarray) -> float:
        try:
            blur = cv2.GaussianBlur(crop_bin, (3, 3), 0)
            score, _ = ssim(crop_bin, blur, full=True)
        except Exception:
            score = float((crop_bin > 0).sum()) / crop_bin.size

        return float(max(0, min(1, score)))

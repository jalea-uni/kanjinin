import json, pathlib, math
from typing import Tuple, Optional
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

class KanjiEvaluator:
    def __init__(self, model_dir: pathlib.Path):
        self.model_dir = model_dir
        self.model = None
        self.labels = None  # two-way mapping
        self.model_version = "unknown"

        self._load_model_safe()

    def _load_model_safe(self):
        model_path = self.model_dir / "model.pth"
        label_map_path = self.model_dir / "label_map.json"
        try:
            if model_path.exists():
                # Placeholder: utente fornisce definizione del modello compatibile con state_dict
                # Qui usiamo un semplice MLP dummy se non viene sostituito.
                self.model = torch.nn.Sequential(
                    torch.nn.Flatten(),
                    torch.nn.Linear(64*64, 256),
                    torch.nn.ReLU(),
                    torch.nn.Linear(256, 128),
                    torch.nn.ReLU(),
                    torch.nn.Linear(128, 128),  # supponiamo 128 classi
                )
                self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
                self.model.eval()
                self.model_version = "user-model"
            else:
                # mock model
                self.model = None
                self.model_version = "mock"
        except Exception as e:
            # fallback mock
            self.model = None
            self.model_version = f"mock ({e})"

        # label map
        self.labels = {}
        if label_map_path.exists():
            with open(label_map_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            # normalizza a due dizionari
            if all(isinstance(k, str) and isinstance(v, int) for k,v in raw.items()):
                forward = raw
                inverse = {v:k for k,v in raw.items()}
            else:
                # forse è {index: char}
                forward = {}
                inverse = {}
                for k,v in raw.items():
                    try:
                        idx = int(k)
                        inverse[idx] = v
                        forward[v] = idx
                    except:
                        pass
            self.labels = {"forward": forward, "inverse": inverse}

    def _crop_to_tensor(self, crop: np.ndarray) -> torch.Tensor:
        # Porta a 64x64 e normalizza [0,1]
        import cv2
        img = crop
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (64,64), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)  # C,H,W
        return torch.from_numpy(img).unsqueeze(0)  # B,C,H,W

    def predict(self, crop: np.ndarray) -> Tuple[Optional[str], float]:
        """Ritorna (predicted_char, confidence). Se mock, stima fittizia."""
        if self.model is None:
            # Mock: usa densità di inchiostro per generare una pseudo-confidenza
            ink = float((crop > 0).sum()) / float(crop.size)
            conf = max(0.1, min(0.99, 1.2*ink))
            # senza label map non possiamo sapere il char → None
            return None, conf
        with torch.no_grad():
            x = self._crop_to_tensor(crop)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            char = self.labels.get("inverse", {}).get(idx, None)
            return char, conf

    def quality(self, crop_bin: np.ndarray) -> float:
        # Proxy semplice: SSIM contro il "miglioramento" (blur leggero) oppure densità di stroke.
        import cv2
        blur = cv2.GaussianBlur(crop_bin, (3,3), 0)
        score, _ = ssim(crop_bin, blur, full=True)
        # Riporta in [0,1]
        return float(max(0.0, min(1.0, score)))

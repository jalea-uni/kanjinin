import io, os, json, uuid, time, math, pathlib, datetime
from typing import Tuple, Dict, Any
from PIL import Image, ImageOps

SESS_ROOT = pathlib.Path(__file__).resolve().parent.parent / "sessions"

def now_session_id() -> str:
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]

def ensure_dir(path: pathlib.Path):
    path.mkdir(parents=True, exist_ok=True)

def save_bytes_to(path: pathlib.Path, data: bytes):
    ensure_dir(path.parent)
    with open(path, "wb") as f:
        f.write(data)

def load_image_fix_exif(content: bytes) -> Image.Image:
    # PIL gestisce l'orientamento EXIF via exif_transpose
    img = Image.open(io.BytesIO(content))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def mm_to_px(mm: float, dpi: int) -> int:
    return int(round(mm * dpi / 25.4))

def a4_size_px(dpi: int) -> Tuple[int, int]:
    # A4: 210mm x 297mm
    return (mm_to_px(210, dpi), mm_to_px(297, dpi))

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))

def try_parse_json(text: str, default: Any) -> Any:
    try:
        return json.loads(text)
    except Exception:
        return default

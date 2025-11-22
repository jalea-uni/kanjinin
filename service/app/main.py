import os, io, json, time, pathlib, uuid
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

import numpy as np
import cv2
from PIL import Image

from . import schemas
from .utils import now_session_id, ensure_dir, load_image_fix_exif, try_parse_json
from .image_pipeline import to_cv, warp_to_a4, deskew, enhance_and_binarize, to_pil
from .layouts import grid_layout_pixels
from .evaluator import KanjiEvaluator

# -------------------------------------------------------------------
# Paths & app setup
# -------------------------------------------------------------------

APP_ROOT = pathlib.Path(__file__).resolve().parent
SESS_ROOT = APP_ROOT.parent / "sessions"
MODELS_DIR = APP_ROOT.parent / "models"

print("[MAIN] MODELS_DIR:", MODELS_DIR, "exists:", MODELS_DIR.exists())

app = FastAPI(title="KanjiNin Image Service", version="0.1.0")

# Valutatore globale
evaluator = KanjiEvaluator(MODELS_DIR)


# -------------------------------------------------------------------
# Healthcheck
# -------------------------------------------------------------------


@app.get("/healthz")
def healthz():
    return {"ok": True, "model": evaluator.model_version}


# -------------------------------------------------------------------
# Evaluate endpoint
# -------------------------------------------------------------------


@app.post("/evaluate", response_model=schemas.EvaluateResponse)
async def evaluate(
    doc: UploadFile = File(..., description="camera image (jpg/png)"),
    kanji_list: str = Form(
        ..., description="JSON array, each item at least has 'kanji'"
    ),
    options: Optional[str] = Form(None, description="JSON options"),
):
    """
    Valuta una pagina di kanji:
    - doc: immagine (foto/scansione)
    - kanji_list: JSON array con almeno la chiave 'kanji' o stringa diretta
    - options: parametri vari (dpi, grid, ecc.)
    """
    t0 = time.time()

    # ---------------------------------------------------------------
    # Validazione input
    # ---------------------------------------------------------------
    if doc.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, f"Unsupported content-type: {doc.content_type}")

    # parse JSON inputs
    try:
        expected_list = json.loads(kanji_list)
        if not isinstance(expected_list, list):
            raise ValueError("kanji_list must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"Invalid kanji_list JSON: {e}")

    opts = try_parse_json(options or "{}", default={})
    dpi = int(opts.get("dpi", 300))

    grid = opts.get("grid", {"rows": 6, "cols": 7})
    rows = int(grid.get("rows", 6))
    cols = int(grid.get("cols", 7))

    box_mm = float(opts.get("box_mm", 30))
    gap_mm = float(opts.get("gap_mm", 10))
    page_margins_mm = opts.get(
        "page_margins_mm",
        {"top": 20, "left": 20, "right": 20, "bottom": 20},
    )
    return_crops = bool(opts.get("return_crops", True))

    # ---------------------------------------------------------------
    # Caricamento & normalizzazione immagine
    # ---------------------------------------------------------------
    raw = await doc.read()
    img = load_image_fix_exif(raw)  # PIL image
    img_cv = to_cv(img)  # BGR OpenCV
    a4 = warp_to_a4(img_cv, dpi=dpi)
    a4 = deskew(a4, max_angle=3.0)
    bin_img = enhance_and_binarize(a4)  # immagine binarizzata

    # ---------------------------------------------------------------
    # Layout griglia (A4, righe/colonne, margini, ecc.)
    # ---------------------------------------------------------------
    bboxes = grid_layout_pixels(
        dpi=dpi,
        rows=rows,
        cols=cols,
        page_margins_mm=page_margins_mm,
        box_mm=box_mm,
        gap_mm=gap_mm,
    )

    # ---------------------------------------------------------------
    # Sessione e cartella crops
    # ---------------------------------------------------------------
    session_id = now_session_id()
    sess_dir = SESS_ROOT / session_id
    crops_dir = sess_dir / "crops"
    ensure_dir(crops_dir)

    # ---------------------------------------------------------------
    # Itera sui box, valuta e costruisci items
    # ---------------------------------------------------------------
    items = []
    qualities = []

    H, W = bin_img.shape[:2]

    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox

        # Bound check
        x1 = max(0, int(x))
        y1 = max(0, int(y))
        x2 = min(W, int(x + w))
        y2 = min(H, int(y + h))

        if x2 <= x1 or y2 <= y1:
            continue

        crop = bin_img[y1:y2, x1:x2].copy()

        # expected kanji
        expected_char = None
        if idx < len(expected_list):
            item = expected_list[idx]
            if isinstance(item, dict):
                expected_char = item.get("kanji")
            elif isinstance(item, str):
                expected_char = item

        # model prediction + quality metric
        pred_char, conf = evaluator.predict(crop)
        q = evaluator.quality(crop)
        qualities.append(q)

        assets = None
        if return_crops:
            crop_name = f"p1-b{idx+1:02d}.png"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)  # binarized crop
            assets = {
                "crop_relpath": str(
                    pathlib.Path("sessions") / session_id / "crops" / crop_name
                )
            }

        # DEBUG opzionale
        # print(f"[EVAL] box {idx} expected={expected_char!r} pred={pred_char!r} conf={conf:.3f}")

        print (pred_char)
        items.append(
            {
                "id": f"p1-b{idx+1:02d}",
                "page": 1,
                "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
                "expected_kanji": expected_char,
                "predicted_kanji": pred_char,
                "confidence": conf,
                "quality": q,
                "assets": assets,
                "notes": None,
            }
        )

    # ---------------------------------------------------------------
    # Statistiche / summary (solo box con kanji atteso)
    # ---------------------------------------------------------------
    ms = int((time.time() - t0) * 1000)

    # Tieni solo i box che hanno effettivamente un kanji atteso
    items = [it for it in items if it.get("expected_kanji")]

    total = len(items)

    if total > 0:
        matched = sum(
            1
            for it in items
            if it.get("predicted_kanji") is not None
            and it["predicted_kanji"] == it["expected_kanji"]
        )
        acc = matched / total
        q_vals = [it.get("quality") for it in items if it.get("quality") is not None]
        avg_q = sum(q_vals) / len(q_vals) if q_vals else None
    else:
        matched = 0
        acc = None
        avg_q = None

    resp = {
        "session_id": session_id,
        "source": {"type": "image", "dpi": dpi, "normalized": "A4"},
        "summary": {
            # ⚠️ ora conta SOLO i riquadri con kanji atteso
            "total_boxes": total,
            "matched": matched,
            "accuracy_top1": acc,
            "avg_quality": avg_q,
            "processing_ms": ms,
            "model_version": evaluator.model_version,
        },
        "items": items,
    }

    return JSONResponse(resp)

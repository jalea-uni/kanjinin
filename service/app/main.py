import os, io, json, time, pathlib, uuid
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
from . import schemas
from .utils import now_session_id, ensure_dir, load_image_fix_exif, try_parse_json
from .image_pipeline import to_cv, warp_to_a4, deskew, enhance_and_binarize, to_pil
from .layouts import grid_layout_pixels
from .evaluator import KanjiEvaluator

APP_ROOT = pathlib.Path(__file__).resolve().parent
SESS_ROOT = APP_ROOT.parent / "sessions"
MODELS_DIR = APP_ROOT.parent / "models"

app = FastAPI(title="KanjiNin Image Service", version="0.1.0")

# Carica valutatore (mock-safe se manca il modello)
evaluator = KanjiEvaluator(MODELS_DIR)

@app.get("/healthz")
def healthz():
    return {"ok": True, "model": evaluator.model_version}

@app.post("/evaluate", response_model=schemas.EvaluateResponse)
async def evaluate(
    doc: UploadFile = File(..., description="camera image (jpg/png)"),
    kanji_list: str = Form(..., description="JSON array, each item at least has 'kanji'"),
    options: Optional[str] = Form(None, description="JSON options")
):
    t0 = time.time()

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
    page_margins_mm = opts.get("page_margins_mm", {"top":20,"left":20,"right":20,"bottom":20})
    return_crops = bool(opts.get("return_crops", True))

    # load & normalize image
    raw = await doc.read()
    img = load_image_fix_exif(raw)
    img_cv = to_cv(img)
    a4 = warp_to_a4(img_cv, dpi=dpi)
    a4 = deskew(a4, max_angle=3.0)
    bin_img = enhance_and_binarize(a4)

    # layout bboxes
    bboxes = grid_layout_pixels(dpi, rows, cols, page_margins_mm, box_mm, gap_mm)

    # session storage
    session_id = now_session_id()
    sess_dir = SESS_ROOT / session_id
    crops_dir = sess_dir / "crops"
    ensure_dir(crops_dir)

    # iterate boxes and evaluate
    items = []
    matched = 0
    qualities = []
    preds_ok = 0

    for idx, bbox in enumerate(bboxes):
        x,y,w,h = bbox
        # Safe bounds
        H, W = bin_img.shape[:2]
        x2 = min(W, x+w); y2 = min(H, y+h)
        x1 = max(0, x); y1 = max(0, y)
        if x2<=x1 or y2<=y1:
            continue
        crop = bin_img[y1:y2, x1:x2].copy()

        expected_char = None
        if idx < len(expected_list):
            item = expected_list[idx]
            if isinstance(item, dict):
                expected_char = item.get("kanji")
            elif isinstance(item, str):
                expected_char = item

        pred_char, conf = evaluator.predict(crop)
        q = evaluator.quality(crop)

        if expected_char is not None and pred_char is not None and pred_char == expected_char:
            preds_ok += 1
        if expected_char is not None:
            matched += 1
        qualities.append(q)

        assets = None
        if return_crops:
            crop_name = f"p1-b{idx+1:02d}.png"
            crop_path = crops_dir / crop_name
            cv2.imwrite(str(crop_path), crop)  # binarized crop
            assets = {"crop_relpath": str(pathlib.Path("sessions")/session_id/"crops"/crop_name)}

        items.append({
            "id": f"p1-b{idx+1:02d}",
            "page": 1,
            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
            "expected_kanji": expected_char,
            "predicted_kanji": pred_char,
            "confidence": conf,
            "quality": q,
            "assets": assets,
            "notes": None
        })

    total = len(bboxes)
    acc = (preds_ok / matched) if matched>0 else None
    avg_q = (sum(qualities)/len(qualities)) if qualities else None
    ms = int((time.time()-t0)*1000)

    items = [it for it in items if it.get("expected_kanji")]

    resp = {
        "session_id": session_id,
        "source": {"type":"image","dpi": dpi, "normalized":"A4"},
        "summary": {
            "total_boxes": total,
            "matched": matched,
            "accuracy_top1": acc,
            "avg_quality": avg_q,
            "processing_ms": ms,
            "model_version": evaluator.model_version
        },
        "items": items
    }
    return JSONResponse(resp)

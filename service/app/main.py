# service/app/main.py
import base64
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
from .evaluator import KanjiEvaluator
from .box_detector_try0 import detect_kanji_boxes, extract_box_content, is_box_empty
from .image_normalizer import normalize_image

# -------------------------------------------------------------------
# CONFIGURAZIONE DEBUG
# -------------------------------------------------------------------
ENABLE_DEBUG = True  # Cambia a False per disabilitare debug completo
DEBUG_SAVE_CROPS = True  # Salva crop individuali

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
    Valuta una pagina di kanji
    """
    t0 = time.time()
    session_id = now_session_id()

    # ---------------------------------------------------------------
    # Setup directories
    # ---------------------------------------------------------------
    sess_dir = SESS_ROOT / session_id
    crops_dir = sess_dir / "crops"
    ensure_dir(crops_dir)

    if ENABLE_DEBUG:
        debug_dir = sess_dir / "debug"
        ensure_dir(debug_dir)
        print(f"\n{'='*60}")
        print(f"[SESSION {session_id}] INIZIO VALUTAZIONE")
        print(f"{'='*60}")

    # ---------------------------------------------------------------
    # Validazione input
    # ---------------------------------------------------------------
    if doc.content_type not in ("image/jpeg", "image/png"):
        raise HTTPException(400, f"Unsupported content-type: {doc.content_type}")

    try:
        expected_list = json.loads(kanji_list)
        if not isinstance(expected_list, list):
            raise ValueError("kanji_list must be a JSON array")
    except Exception as e:
        raise HTTPException(400, f"Invalid kanji_list JSON: {e}")

    if ENABLE_DEBUG:
        print(f"[INPUT] Content-Type: {doc.content_type}")
        print(f"[INPUT] Expected kanji count: {len(expected_list)}")
        print(
            f"[INPUT] Expected kanji: {[k.get('kanji') if isinstance(k, dict) else k for k in expected_list[:5]]}..."
        )

    opts = try_parse_json(options or "{}", default={})
    dpi = int(opts.get("dpi", 300))
    return_crops = bool(opts.get("return_crops", True))

    # ---------------------------------------------------------------
    # Caricamento immagine
    # ---------------------------------------------------------------
    raw = await doc.read()

    if ENABLE_DEBUG:
        raw_path = debug_dir / "01_raw_input.png"
        with open(raw_path, "wb") as f:
            f.write(raw)
        print(f"[DEBUG] ✓ RAW salvato: {len(raw)} bytes")

    img = load_image_fix_exif(raw)
    img_cv = to_cv(img)

    if ENABLE_DEBUG:
        cv2.imwrite(str(debug_dir / "02_after_to_cv.png"), img_cv)
        print(f"[DEBUG] ✓ to_cv: shape={img_cv.shape}")

    # ---------------------------------------------------------------
    # NORMALIZZAZIONE (deskew automatico)
    # ---------------------------------------------------------------
    if ENABLE_DEBUG:
        print(f"\n[NORMALIZE] Analisi rotazione...")

    img_cv, normalize_info = normalize_image(img_cv, max_rotation=45.0)

    if ENABLE_DEBUG:
        print(f"[NORMALIZE] Angolo rilevato: {normalize_info['angle_detected']:.1f}°")
        print(f"[NORMALIZE] Ruotato: {normalize_info['was_rotated']}")
        if normalize_info["was_rotated"]:
            cv2.imwrite(str(debug_dir / "02b_after_normalize.png"), img_cv)

    # ---------------------------------------------------------------
    # RILEVAMENTO 1: SENZA PREPROCESSING
    # ---------------------------------------------------------------
    gray_direct = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    if ENABLE_DEBUG:
        cv2.imwrite(str(debug_dir / "03_gray_direct.png"), gray_direct)
        print(f"\n[TEST] Rilevamento SENZA preprocessing...")

    bboxes_no_preproc = detect_kanji_boxes(
        gray_direct,
        min_area=2000,  # Più basso per trovare box piccoli
        max_area=100000,  # Più alto per sicurezza
    )

    if ENABLE_DEBUG:
        print(f"[TEST] Box trovati: {len(bboxes_no_preproc)}")
        if len(bboxes_no_preproc) > 0:
            debug_img = cv2.cvtColor(gray_direct, cv2.COLOR_GRAY2BGR)
            for idx, (x, y, w, h) in enumerate(bboxes_no_preproc):
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(
                    debug_img,
                    f"{idx}",
                    (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            cv2.imwrite(str(debug_dir / "04_boxes_no_preproc.png"), debug_img)

    # ---------------------------------------------------------------
    # RILEVAMENTO 2: CON PREPROCESSING
    # ---------------------------------------------------------------
    if ENABLE_DEBUG:
        print(f"\n[PREPROC] Applicazione warp_to_a4...")

    a4 = warp_to_a4(img_cv, dpi=dpi)

    if ENABLE_DEBUG:
        cv2.imwrite(str(debug_dir / "05_after_warp_a4.png"), a4)
        print(f"[PREPROC] Applicazione deskew...")

    a4_deskewed = deskew(a4, max_angle=3.0)

    if ENABLE_DEBUG:
        cv2.imwrite(str(debug_dir / "06_after_deskew.png"), a4_deskewed)

    gray_preprocessed = cv2.cvtColor(a4_deskewed, cv2.COLOR_BGR2GRAY)

    if ENABLE_DEBUG:
        cv2.imwrite(str(debug_dir / "07_gray_preprocessed.png"), gray_preprocessed)
        print(f"\n[TEST] Rilevamento CON preprocessing...")

    bboxes_with_preproc = detect_kanji_boxes(
        gray_preprocessed, min_area=2000, max_area=100000
    )

    if ENABLE_DEBUG:
        print(f"[TEST] Box trovati: {len(bboxes_with_preproc)}")
        if len(bboxes_with_preproc) > 0:
            debug_img = cv2.cvtColor(gray_preprocessed, cv2.COLOR_GRAY2BGR)
            for idx, (x, y, w, h) in enumerate(bboxes_with_preproc):
                cv2.rectangle(debug_img, (x, y), (x + w, y + h), (255, 0, 0), 3)
                cv2.putText(
                    debug_img,
                    f"{idx}",
                    (x + 5, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    2,
                )
            cv2.imwrite(str(debug_dir / "08_boxes_with_preproc.png"), debug_img)

    # ---------------------------------------------------------------
    # DECISIONE: USA IL METODO MIGLIORE
    # ---------------------------------------------------------------
    if len(bboxes_no_preproc) >= len(bboxes_with_preproc):
        if ENABLE_DEBUG:
            print(
                f"\n[DECISIONE] Uso SENZA preprocessing ({len(bboxes_no_preproc)} box)"
            )

        bboxes = bboxes_no_preproc
        # ⚠️ IMPORTANTE: Binarizza l'immagine NON preprocessata
        bin_img = enhance_and_binarize(img_cv)

        if ENABLE_DEBUG:
            cv2.imwrite(str(debug_dir / "09_binarized_no_preproc.png"), bin_img)

        use_preprocessed = False
    else:
        if ENABLE_DEBUG:
            print(
                f"\n[DECISIONE] Uso CON preprocessing ({len(bboxes_with_preproc)} box)"
            )

        bboxes = bboxes_with_preproc
        # Binarizza l'immagine preprocessata
        bin_img = enhance_and_binarize(a4_deskewed)

        if ENABLE_DEBUG:
            cv2.imwrite(str(debug_dir / "09_binarized_with_preproc.png"), bin_img)

        use_preprocessed = True

    # ---------------------------------------------------------------
    # ESTRAZIONE E VALUTAZIONE
    # ---------------------------------------------------------------
    items = []
    qualities = []

    if ENABLE_DEBUG:
        print(f"\n[VALUTAZIONE] Elaborazione {len(bboxes)} box...")

    for idx, bbox in enumerate(bboxes):
        x, y, w, h = bbox

        # Estrai contenuto del box
        crop = extract_box_content(bin_img, bbox, padding=10)

        # Skip se vuoto
        if crop is None or is_box_empty(crop):
            if ENABLE_DEBUG:
                print(f"[BOX {idx}] VUOTO - skip")
            continue

        # Salva crop per debug
        if ENABLE_DEBUG and DEBUG_SAVE_CROPS:
            cv2.imwrite(str(debug_dir / f"box_{idx:02d}_crop.png"), crop)

        # Expected kanji
        expected_char = None
        if idx < len(expected_list):
            item = expected_list[idx]
            if isinstance(item, dict):
                expected_char = item.get("kanji")
            elif isinstance(item, str):
                expected_char = item

        # Model prediction
        pred_char, conf = evaluator.predict(crop)
        q = evaluator.quality(crop)
        qualities.append(q)

        # Assets per frontend
        assets = None
        if return_crops:
            crop_name_final = f"p1-b{idx+1:02d}.png"
            crop_path = crops_dir / crop_name_final
            cv2.imwrite(str(crop_path), crop)
            _, buffer = cv2.imencode('.png', crop)
            crop_base64 = base64.b64encode(buffer).decode('utf-8')
            assets = {
                "crop_relpath": str(
                    pathlib.Path("sessions") / session_id / "crops" / crop_name_final
                ),
                "crop_base64": crop_base64
            }

        # Log risultato
        if ENABLE_DEBUG:
            status = "✓" if pred_char == expected_char else "✗"
            print(
                f"[BOX {idx}] {status} expected={expected_char!r} predicted={pred_char!r} conf={conf:.2f}"
            )

        items.append(
            {
                "id": f"p1-b{idx+1:02d}",
                "page": 1,
                "bbox": [int(x), int(y), int(w), int(h)],
                "expected_kanji": expected_char,
                "predicted_kanji": pred_char,
                "confidence": conf,
                "quality": q,
                "assets": assets,
                "notes": None,
            }
        )

    # ---------------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------------
    ms = int((time.time() - t0) * 1000)

    # Filtra solo box con kanji atteso
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

    if ENABLE_DEBUG:
        print(f"\n{'='*60}")
        print(f"[SUMMARY] Box valutati: {total}")
        print(f"[SUMMARY] Match corretti: {matched}/{total}")
        print(f"[SUMMARY] Accuracy: {acc*100 if acc else 0:.1f}%")
        print(f"[SUMMARY] Tempo: {ms}ms")
        print(f"[SUMMARY] Debug: {debug_dir}")
        print(f"{'='*60}\n")

    resp = {
        "session_id": session_id,
        "source": {
            "type": "image",
            "dpi": dpi,
            "normalized": "A4" if use_preprocessed else "direct",
        },
        "summary": {
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

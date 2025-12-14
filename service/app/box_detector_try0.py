# service/app/box_detector.py - VERSIONE SEMPLIFICATA

import cv2
import numpy as np
from typing import List, Tuple


def detect_kanji_boxes(
    img: np.ndarray, min_area: int = 8000, max_area: int = 50000
) -> List[Tuple[int, int, int, int]]:
    """
    Rileva box kanji con approccio semplificato
    """

    # Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

    # Dilata moderatamente
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Contorni
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        # Approssima poligono
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # DEVE essere un quadrilatero (4 vertici)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(approx)
        area = w * h

        # Filtra per area
        if area < min_area or area > max_area:
            continue

        # Aspect ratio quasi quadrato
        aspect = w / h if h > 0 else 0
        if aspect < 0.7 or aspect > 1.4:
            continue

        boxes.append((x, y, w, h))

    # Rimuovi sovrapposizioni
    boxes = remove_overlapping_boxes(boxes, iou_threshold=0.4)

    # Ordina
    boxes = sorted(boxes, key=lambda b: (b[1] // 50, b[0]))

    return boxes


def remove_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.4
) -> List[Tuple[int, int, int, int]]:
    """Rimuove box sovrapposti"""
    if len(boxes) == 0:
        return []

    boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)

    keep = []
    for box in boxes:
        x1, y1, w1, h1 = box

        overlaps = False
        for kept_box in keep:
            x2, y2, w2, h2 = kept_box

            x_left = max(x1, x2)
            y_top = max(y1, y2)
            x_right = min(x1 + w1, x2 + w2)
            y_bottom = min(y1 + h1, y2 + h2)

            if x_right > x_left and y_bottom > y_top:
                intersection = (x_right - x_left) * (y_bottom - y_top)
                area1 = w1 * h1
                iou = intersection / area1

                if iou > iou_threshold:
                    overlaps = True
                    break

        if not overlaps:
            keep.append(box)

    return keep


def extract_box_content(
    img: np.ndarray, bbox: Tuple[int, int, int, int], padding: int = 10
) -> np.ndarray:
    """Estrae contenuto del box"""
    x, y, w, h = bbox

    x_start = max(0, x + padding)
    y_start = max(0, y + padding)
    x_end = min(img.shape[1], x + w - padding)
    y_end = min(img.shape[0], y + h - padding)

    if x_end <= x_start or y_end <= y_start:
        return None

    return img[y_start:y_end, x_start:x_end]


def is_box_empty(box_image: np.ndarray, ink_threshold: float = 0.02) -> bool:
    """Verifica se box è vuoto"""
    if box_image is None or box_image.size == 0:
        return True

    if box_image.ndim == 3:
        gray = cv2.cvtColor(box_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = box_image

    ink_ratio = float(np.sum(gray < 128)) / gray.size

    return ink_ratio < ink_threshold

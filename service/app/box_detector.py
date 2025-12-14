# service/app/box_detector.py - CON GRID SPLITTING

import cv2
import numpy as np
from typing import List, Tuple


def detect_kanji_boxes(
    img: np.ndarray, min_area: int = 2000, max_area: int = 100000
) -> List[Tuple[int, int, int, int]]:
    """
    1. Trova i box GRANDI (righe/gruppi)
    2. Suddividi ogni box grande in celle individuali
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

    # Dilata
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Trova contorni GRANDI
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    large_boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Box GRANDI (almeno 20000 pixel)
        if area < 20000:
            continue

        large_boxes.append((x, y, w, h))

    print(f"[DEBUG] Box grandi trovati: {len(large_boxes)}")

    # ✅ SUDDIVIDI ogni box grande in celle
    individual_boxes = []

    for large_x, large_y, large_w, large_h in large_boxes:
        # Estrai regione
        roi = gray[large_y : large_y + large_h, large_x : large_x + large_w]

        # Trova linee VERTICALI dentro il box
        v_lines = find_vertical_lines(roi)

        # Se non trova linee, prova a dividere geometricamente
        if len(v_lines) < 2:
            # Dividi in base alla larghezza (stima numero celle)
            estimated_cells = max(1, round(large_w / 100))  # 1 cella ogni ~100px
            cell_width = large_w // estimated_cells

            v_lines = [i * cell_width for i in range(estimated_cells + 1)]

        print(f"[DEBUG] Box ({large_x},{large_y}) -> {len(v_lines)-1} celle")

        # Crea box individuali dalle linee verticali
        for i in range(len(v_lines) - 1):
            x1 = large_x + v_lines[i]
            x2 = large_x + v_lines[i + 1]

            cell_w = x2 - x1
            cell_h = large_h

            # Filtra celle troppo piccole o troppo grandi
            cell_area = cell_w * cell_h
            if cell_area < min_area or cell_area > max_area:
                continue

            # Aspect ratio ragionevole
            aspect = cell_w / cell_h if cell_h > 0 else 0
            if aspect < 0.4 or aspect > 2.5:
                continue

            individual_boxes.append((x1, large_y, cell_w, cell_h))

    print(f"[DEBUG] Box individuali creati: {len(individual_boxes)}")

    # Rimuovi sovrapposizioni
    individual_boxes = remove_overlapping_boxes(individual_boxes, iou_threshold=0.5)

    # Ordina
    individual_boxes = sorted(individual_boxes, key=lambda b: (b[1] // 100, b[0]))

    return individual_boxes


def find_vertical_lines(roi: np.ndarray) -> List[int]:
    """
    Trova le linee verticali in una ROI usando proiezione orizzontale
    """
    h, w = roi.shape

    # Inverti per avere bordi bianchi
    inverted = 255 - roi

    # Threshold
    _, binary = cv2.threshold(inverted, 50, 255, cv2.THRESH_BINARY)

    # Proiezione verticale (somma lungo le colonne)
    projection = np.sum(binary, axis=0)

    # Trova picchi (dove ci sono linee verticali)
    threshold = h * 50  # Soglia: almeno 50 pixel di altezza

    lines = []
    in_peak = False
    peak_start = 0

    for i, val in enumerate(projection):
        if val > threshold and not in_peak:
            in_peak = True
            peak_start = i
        elif val <= threshold and in_peak:
            # Fine picco - prendi il centro
            lines.append((peak_start + i) // 2)
            in_peak = False

    # Aggiungi bordi
    if 0 not in lines:
        lines.insert(0, 0)
    if w not in lines:
        lines.append(w)

    return sorted(lines)


def remove_overlapping_boxes(
    boxes: List[Tuple[int, int, int, int]], iou_threshold: float = 0.5
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

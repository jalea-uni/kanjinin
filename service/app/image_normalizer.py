# service/app/image_normalizer.py
# Modulo separato per normalizzazione immagini storte

import cv2
import numpy as np
from typing import Tuple, Optional


def normalize_image(
    img: np.ndarray, max_rotation: float = 45.0
) -> Tuple[np.ndarray, dict]:
    """
    Normalizza un'immagine storta:
    1. Rileva l'angolo di rotazione
    2. Corregge la rotazione
    3. Ritorna immagine corretta + info

    Args:
        img: Immagine BGR
        max_rotation: Angolo massimo da correggere (default 45°)

    Returns:
        (immagine_corretta, info_dict)
    """
    info = {"angle_detected": 0.0, "was_rotated": False, "method": "none"}

    # Grayscale
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Prova diversi metodi per trovare l'angolo
    angle = detect_rotation_angle(gray, max_rotation)

    info["angle_detected"] = angle

    # Se l'angolo è significativo (> 0.5°), correggi
    if abs(angle) > 0.5:
        img = rotate_image(img, -angle)  # Ruota in senso opposto
        info["was_rotated"] = True
        info["method"] = "auto_deskew"

    return img, info


def detect_rotation_angle(gray: np.ndarray, max_angle: float = 45.0) -> float:
    """
    Rileva l'angolo di rotazione usando multiple tecniche
    """
    angles = []

    # Metodo 1: Hough Lines
    angle_hough = detect_angle_hough(gray)
    if angle_hough is not None and abs(angle_hough) <= max_angle:
        angles.append(angle_hough)

    # Metodo 2: minAreaRect sul contenuto
    angle_rect = detect_angle_min_rect(gray)
    if angle_rect is not None and abs(angle_rect) <= max_angle:
        angles.append(angle_rect)

    # Metodo 3: Proiezione (per angoli piccoli)
    angle_proj = detect_angle_projection(gray)
    if angle_proj is not None and abs(angle_proj) <= max_angle:
        angles.append(angle_proj)

    if not angles:
        return 0.0

    # Prendi la mediana degli angoli trovati
    return float(np.median(angles))


def detect_angle_hough(gray: np.ndarray) -> Optional[float]:
    """
    Rileva angolo usando Hough Lines Transform
    """
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Hough Lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None or len(lines) == 0:
        return None

    angles = []
    for line in lines:
        rho, theta = line[0]
        # Converti theta in gradi
        angle_deg = np.degrees(theta) - 90

        # Normalizza a range [-45, 45]
        if angle_deg < -45:
            angle_deg += 90
        elif angle_deg > 45:
            angle_deg -= 90

        angles.append(angle_deg)

    if not angles:
        return None

    # Mediana per robustezza
    return float(np.median(angles))


def detect_angle_min_rect(gray: np.ndarray) -> Optional[float]:
    """
    Rileva angolo usando il rettangolo minimo del contenuto
    """
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Trova contorni
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # Unisci tutti i contorni
    all_points = np.vstack(contours)

    # Trova rettangolo minimo
    rect = cv2.minAreaRect(all_points)
    angle = rect[2]

    # Normalizza angolo
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    return float(angle)


def detect_angle_projection(
    gray: np.ndarray, angle_range: float = 15.0
) -> Optional[float]:
    """
    Rileva angolo usando proiezione orizzontale
    Efficace per angoli piccoli (< 15°)
    """
    # Threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle = 0.0
    best_score = 0.0

    # Prova angoli da -angle_range a +angle_range
    for angle in np.arange(-angle_range, angle_range + 0.5, 0.5):
        rotated = rotate_image(binary, angle)

        # Proiezione orizzontale
        projection = np.sum(rotated, axis=1)

        # Score = varianza della proiezione (più alta = più allineato)
        score = np.var(projection)

        if score > best_score:
            best_score = score
            best_angle = angle

    return best_angle


def rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    Ruota l'immagine di un angolo specifico mantenendo tutto il contenuto
    """
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    # Matrice di rotazione
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # Calcola nuove dimensioni per contenere tutta l'immagine ruotata
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Aggiusta la matrice per centrare
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Ruota con sfondo bianco
    if img.ndim == 3:
        border_color = (255, 255, 255)
    else:
        border_color = 255

    rotated = cv2.warpAffine(
        img,
        M,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color,
    )

    return rotated

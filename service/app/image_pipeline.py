import numpy as np
import cv2
from PIL import Image
from typing import Tuple
from .utils import a4_size_px

def to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def to_pil(img_cv: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

def detect_page_quad(img_cv: np.ndarray):
    # Contour più grande come pagina. Fallback: tutta l'immagine.
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape[:2]
        return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    cnt = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
    if len(approx) == 4:
        quad = approx.reshape(4,2).astype(np.float32)
        # Ordina i punti TL, TR, BR, BL
        s = quad.sum(axis=1)
        diff = np.diff(quad, axis=1).reshape(-1)
        tl = quad[np.argmin(s)]
        br = quad[np.argmax(s)]
        tr = quad[np.argmin(diff)]
        bl = quad[np.argmax(diff)]
        return np.array([tl,tr,br,bl], dtype=np.float32)
    # fallback rettangolo pieno
    h, w = gray.shape[:2]
    return np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)

def warp_to_a4(img_cv: np.ndarray, dpi: int):
    W, H = a4_size_px(dpi)
    quad = detect_page_quad(img_cv)
    dst = np.array([[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(quad, dst)
    warped = cv2.warpPerspective(img_cv, M, (W, H), flags=cv2.INTER_CUBIC)
    return warped

def deskew(img_cv: np.ndarray, max_angle: float = 3.0):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=200)
    if lines is None:
        return img_cv
    angles = []
    for rho, theta in lines[:,0]:
        ang = (theta * 180/np.pi) - 90
        if -max_angle <= ang <= max_angle:
            angles.append(ang)
    if not angles:
        return img_cv
    angle = np.median(angles)
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_cv, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def enhance_and_binarize(img_cv: np.ndarray):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    # Adaptive threshold
    bin_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 35, 10)
    # Denoise leggero
    bin_img = cv2.medianBlur(bin_img, 3)
    return bin_img

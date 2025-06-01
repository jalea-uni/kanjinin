import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont


def segment_characters(image, thresh=127, min_area=100):
    # image: numpy array (grayscale or color)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c)>min_area]
    boxes = sorted(boxes, key=lambda b: b[0])
    return [gray[y:y+h, x:x+w] for (x, y, w, h) in boxes]


def compute_ssim(img1, img2):
    # expects 2D numpy arrays of same size
    score, _ = ssim(img1, img2, full=True)
    return score


def render_template(char_code, size=(64, 64), font_path=None):
    img = Image.new('L', size, color=255)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, size=int(size[1]*0.8)) if font_path else ImageFont.load_default()
    except IOError:
        font = ImageFont.load_default()
    text = chr(char_code)
    
    # Handle both old and new Pillow versions
    try:
        # New Pillow version (>= 8.0.0)
        bbox = draw.textbbox((0, 0), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        # Old Pillow version
        w, h = draw.textsize(text, font=font)
    
    pos = ((size[0]-w)//2, (size[1]-h)//2)
    draw.text(pos, text, fill=0, font=font)
    return np.array(img)

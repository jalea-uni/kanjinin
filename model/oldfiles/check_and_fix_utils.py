#!/usr/bin/env python3
"""
Check and fix the utils.py import issue
"""

import os
import shutil

def check_symlinks():
    """Check the status of symlinks"""
    print("Checking symlinks and files...\n")
    
    files_to_check = [
        'inference/utils.py',
        'inference/model.py',
        'shared/utils.py',
        'shared/model.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            if os.path.islink(file_path):
                target = os.readlink(file_path)
                real_path = os.path.realpath(file_path)
                exists = os.path.exists(real_path)
                print(f"✓ {file_path} -> {target} {'(EXISTS)' if exists else '(BROKEN)'}")
            else:
                print(f"✓ {file_path} (regular file)")
        else:
            print(f"✗ {file_path} (NOT FOUND)")
    print()

def create_complete_utils():
    """Create a complete utils.py with all functions"""
    
    complete_utils_content = '''import cv2
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
'''
    
    return complete_utils_content

def fix_utils_files():
    """Fix utils.py in all locations"""
    print("Fixing utils.py files...\n")
    
    utils_content = create_complete_utils()
    
    # Write to shared/utils.py first
    shared_utils = 'shared/utils.py'
    if os.path.exists('shared'):
        print(f"Writing to {shared_utils}...")
        with open(shared_utils, 'w') as f:
            f.write(utils_content)
        print(f"✓ Updated {shared_utils}")
    
    # Check if inference/utils.py is a symlink
    inference_utils = 'inference/utils.py'
    if os.path.exists(inference_utils):
        if os.path.islink(inference_utils):
            print(f"\n{inference_utils} is a symlink")
            # Remove the symlink and create a real file
            os.remove(inference_utils)
            print(f"Removed symlink {inference_utils}")
        
        # Create a real file
        print(f"Creating real file {inference_utils}...")
        with open(inference_utils, 'w') as f:
            f.write(utils_content)
        print(f"✓ Created {inference_utils}")
    
    # Also check training/utils.py if it exists
    training_utils = 'training/utils.py'
    if os.path.exists(training_utils):
        if os.path.islink(training_utils):
            os.remove(training_utils)
        with open(training_utils, 'w') as f:
            f.write(utils_content)
        print(f"✓ Updated {training_utils}")

def test_imports():
    """Test if imports work now"""
    print("\nTesting imports...")
    
    try:
        os.chdir('inference')
        exec('from utils import render_template, compute_ssim, segment_characters')
        print("✓ Import from inference directory works!")
        os.chdir('..')
    except Exception as e:
        print(f"✗ Import failed: {e}")
        os.chdir('..')
        return False
    
    return True

def main():
    print("Checking and fixing utils.py import issues...\n")
    
    # Check we're in the model directory
    if not os.path.exists('inference'):
        print("Error: Please run this script from the model directory")
        return
    
    # Check current state
    check_symlinks()
    
    # Fix utils files
    fix_utils_files()
    
    # Test imports
    if test_imports():
        print("\n✓ All fixes applied successfully!")
        print("\nNow you can run the test:")
        print("  cd inference")
        print("  python3 test_system.py")
    else:
        print("\n⚠ Import test failed, but files were updated.")
        print("Try running the test anyway.")

if __name__ == "__main__":
    main()
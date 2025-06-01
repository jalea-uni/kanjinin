#!/usr/bin/env python3
"""
Fix test PDF generation to include actual kanji and fix model paths
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def update_test_system_pdf_generation():
    """Update the create_test_pdf function in test_system.py"""
    
    new_create_test_pdf = '''def create_test_pdf(output_path: str):
    """Create a test PDF with some handwritten-style kanji"""
    print(f"Creating test PDF at {output_path}...")
    img_width, img_height = 2480, 3508  # A4 at 300 DPI
    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)
    
    box_size = 200
    margin = 100
    boxes_per_row = 8
    
    # Draw grid
    for row in range(5):
        for col in range(boxes_per_row):
            x = margin + col * (box_size + 50)
            y = margin + row * (box_size + 50)
            draw.rectangle([x, y, x + box_size, y + box_size], outline='black', width=2)
    
    # Try to find a font
    font = None
    font_paths = [
        "/usr/share/fonts/truetype/fonts-japanese-gothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc",  # macOS
        "C:/Windows/Fonts/msgothic.ttc",  # Windows
        os.path.join(MODEL_DIR, "NotoSansCJK-Regular.ttc"),  # Local copy
    ]
    
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 150)
                print(f"Using font: {font_path}")
                break
            except:
                continue
    
    if font is None:
        print("Warning: No Japanese font found. Kanji may not display correctly.")
        font = ImageFont.load_default()
    
    # Add kanji to boxes
    test_kanji = ['活', '役', '期', '待', '具']
    for i, kanji in enumerate(test_kanji):
        row = i // boxes_per_row
        col = i % boxes_per_row
        
        # Calculate position to center the kanji in the box
        x = margin + col * (box_size + 50)
        y = margin + row * (box_size + 50)
        
        # Get text size for centering
        bbox = draw.textbbox((0, 0), kanji, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Center the text in the box
        text_x = x + (box_size - text_width) // 2
        text_y = y + (box_size - text_height) // 2
        
        # Add slight random variation to simulate handwriting
        text_x += np.random.randint(-5, 5)
        text_y += np.random.randint(-5, 5)
        
        # Draw the kanji
        draw.text((text_x, text_y), kanji, fill='black', font=font)
    
    # Save as PDF
    img.save(output_path, "PDF", resolution=300.0)
    print(f"Test PDF created: {output_path}")'''
    
    # Read test_system.py
    test_system_path = 'inference/test_system.py'
    with open(test_system_path, 'r') as f:
        content = f.read()
    
    # Find and replace the create_test_pdf function
    import re
    pattern = r'def create_test_pdf\(output_path: str\):.*?print\(f"Test PDF created: \{output_path\}"\)'
    content = re.sub(pattern, new_create_test_pdf, content, flags=re.DOTALL)
    
    # Write back
    with open(test_system_path, 'w') as f:
        f.write(content)
    
    print("✓ Updated PDF generation in test_system.py")

def fix_evaluate_student_kanji_defaults():
    """Fix default paths in evaluate_student_kanji.py"""
    
    file_path = 'inference/evaluate_student_kanji.py'
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update default paths to be relative to the model directory
    replacements = [
        ("default='best_model.pth'", "default='../models/best_model.pth'"),
        ("default='label_map.json'", "default='../models/label_map.json'"),
    ]
    
    for old, new in replacements:
        content = content.replace(old, new)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print("✓ Fixed default paths in evaluate_student_kanji.py")

def download_font_instruction():
    """Print instructions for downloading a font if needed"""
    
    print("\n" + "="*60)
    print("FONT INSTALLATION")
    print("="*60)
    print("\nIf the test PDF shows empty boxes, you need a Japanese font.")
    print("\nOption 1 - Download Noto Sans CJK:")
    print("  cd /var/www/develop/kanjinin/model")
    print("  wget https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc")
    print("\nOption 2 - Install system fonts:")
    print("  Ubuntu/Debian: sudo apt-get install fonts-noto-cjk")
    print("  macOS: Font should be available in system")
    print("  Windows: Download from Google Fonts")
    print("="*60 + "\n")

def main():
    print("Fixing test PDF generation and model paths...\n")
    
    # Check we're in the right directory
    if not os.path.exists('inference'):
        print("Error: Please run this script from the model directory")
        return
    
    # Apply fixes
    update_test_system_pdf_generation()
    fix_evaluate_student_kanji_defaults()
    
    # Show font instructions
    download_font_instruction()
    
    print("\n✓ Fixes applied!")
    print("\nNow you can run the test again:")
    print("  cd inference")
    print("  python3 test_system.py")

if __name__ == "__main__":
    main()
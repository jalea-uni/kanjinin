#!/usr/bin/env python3
"""
Unified test script that can use manual test files or generate automatic ones
"""

import os
import sys
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import json
import subprocess

# ================== CONFIGURATION ==================
# Set to True to use manual files, False to generate automatic test
USE_MANUAL_FILES = True

# Manual file names (in data/test directory)
MANUAL_PDF_NAME = "kanji_sheet_manual.pdf"
MANUAL_JSON_NAME = "kanji_list_manual.json"

# Automatic test file names
AUTO_PDF_NAME = "kanji_sheet_auto.pdf"
AUTO_JSON_NAME = "kanji_list_auto.json"
# ================================================

# Directory paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'models'))
DATA_TEST_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'data', 'test'))
OUTPUT_TEST_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'output', 'test'))

def check_dependencies():
    """Check if all dependencies are installed"""
    print("\nChecking dependencies...")
    
    # Check Python packages
    packages = ['cv2', 'PIL', 'torch', 'pdf2image', 'numpy', 'torchvision']
    missing = []
    
    for package in packages:
        try:
            if package == 'PIL':
                __import__('PIL')
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"❌ {package} NOT installed")
            missing.append(package)
    
    # Check system dependencies
    try:
        subprocess.run(['pdfinfo', '-v'], capture_output=True, check=True)
        print("✓ poppler-utils installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ poppler-utils NOT installed - run: sudo apt-get install poppler-utils")
        missing.append("poppler-utils")
    
    return len(missing) == 0

def create_test_pdf(output_path: str):
    """Create an automatic test PDF with kanji"""
    print(f"Creating automatic test PDF at {output_path}...")
    
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
        "/usr/share/fonts/opentype/ipafont-gothic/ipagp.ttf",  # Alternative
        "/usr/share/fonts/truetype/takao-gothic/TakaoPGothic.ttf",  # Alternative
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
        print("Warning: No Japanese font found. Trying to download...")
        # Try to download a font
        font_url = "https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc"
        local_font_path = os.path.join(MODEL_DIR, "NotoSansCJK-Regular.ttc")
        
        if not os.path.exists(local_font_path):
            print(f"Downloading font to {local_font_path}...")
            try:
                import urllib.request
                urllib.request.urlretrieve(font_url, local_font_path)
                font = ImageFont.truetype(local_font_path, 150)
                print("Font downloaded successfully!")
            except Exception as e:
                print(f"Could not download font: {e}")
                font = ImageFont.load_default()
        else:
            try:
                font = ImageFont.truetype(local_font_path, 150)
            except:
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
        try:
            bbox = draw.textbbox((0, 0), kanji, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except:
            # Fallback for older PIL versions
            text_width, text_height = draw.textsize(kanji, font=font)
        
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
    print(f"Automatic test PDF created: {output_path}")

def create_test_json(output_path: str):
    """Create automatic test kanji list"""
    test_kanji_list = [
        {"kanji": "活", "furigana": "かつ", "unicode": "U+6D3B"},
        {"kanji": "役", "furigana": "やく", "unicode": "U+5F79"},
        {"kanji": "期", "furigana": "き", "unicode": "U+671F"},
        {"kanji": "待", "furigana": "たい", "unicode": "U+5F85"},
        {"kanji": "具", "furigana": "ぐ", "unicode": "U+5177"}
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(test_kanji_list, f, ensure_ascii=False, indent=2)
    
    print(f"Automatic test JSON created: {output_path}")

def test_model_loading():
    """Test if the model can be loaded"""
    print("\nTesting model loading...")
    
    model_path = os.path.join(MODELS_DIR, 'best_model.pth')
    label_map_path = os.path.join(MODELS_DIR, 'label_map.json')
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return False
    
    if not os.path.exists(label_map_path):
        print(f"❌ Label map file not found: {label_map_path}")
        return False
    
    try:
        from kanji_evaluator import KanjiEvaluator
        evaluator = KanjiEvaluator(
            model_path=model_path,
            label_map_path=label_map_path
        )
        print("✓ Model loaded successfully!")
        print(f"  - Device: {evaluator.device}")
        print(f"  - Number of classes: {evaluator.num_classes}")
        return True
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_pdf_extraction():
    """Test PDF box extraction"""
    print("\nTesting PDF box extraction...")
    
    try:
        from pdf_box_extractor import PDFBoxExtractor
        extractor = PDFBoxExtractor()
        
        test_img = np.ones((1000, 1000, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_img, (100, 100), (300, 300), (0, 0, 0), 2)
        cv2.rectangle(test_img, (400, 100), (600, 300), (0, 0, 0), 2)
        
        boxes = extractor.detect_boxes(test_img)
        print(f"✓ Box detection working! Found {len(boxes)} boxes")
        return True
    except Exception as e:
        print(f"❌ Error in PDF extraction: {e}")
        return False

def test_complete_pipeline():
    """Test the complete pipeline with either manual or automatic files"""
    print(f"\nTesting complete pipeline (mode: {'MANUAL' if USE_MANUAL_FILES else 'AUTOMATIC'})...")
    
    # Ensure directories exist
    os.makedirs(DATA_TEST_DIR, exist_ok=True)
    os.makedirs(OUTPUT_TEST_DIR, exist_ok=True)
    
    # Determine which files to use
    if USE_MANUAL_FILES:
        test_pdf_name = MANUAL_PDF_NAME
        test_json_name = MANUAL_JSON_NAME
        print("Using manual test files...")
    else:
        test_pdf_name = AUTO_PDF_NAME
        test_json_name = AUTO_JSON_NAME
        print("Using automatic test files...")
    
    test_pdf_path = os.path.join(DATA_TEST_DIR, test_pdf_name)
    test_list_path = os.path.join(DATA_TEST_DIR, test_json_name)
    
    # For automatic mode, create the files
    if not USE_MANUAL_FILES:
        create_test_json(test_list_path)
        create_test_pdf(test_pdf_path)
    else:
        # For manual mode, check if files exist
        if not os.path.exists(test_pdf_path):
            print(f"❌ Manual PDF not found: {test_pdf_path}")
            print("  Please create this file or switch to automatic mode (USE_MANUAL_FILES = False)")
            return False
        
        if not os.path.exists(test_list_path):
            print(f"❌ Manual JSON not found: {test_list_path}")
            print("  Please create this file or switch to automatic mode (USE_MANUAL_FILES = False)")
            return False
        
        print(f"✓ Found manual PDF: {test_pdf_path}")
        print(f"✓ Found manual JSON: {test_list_path}")
    
    # Show what's in the kanji list
    with open(test_list_path, 'r', encoding='utf-8') as f:
        kanji_list = json.load(f)
    print(f"  Test includes {len(kanji_list)} kanji: {[k['kanji'] for k in kanji_list[:5]]}...")
    
    # Run evaluation
    print("\nRunning evaluation...")
    evaluator_script = os.path.join(BASE_DIR, 'evaluate_student_kanji.py')
    
    cmd = (
        f'"{sys.executable}" "{evaluator_script}" '
        f'"{test_pdf_path}" --kanji-list "{test_list_path}" '
        f'--model-path "{os.path.join(MODELS_DIR, "best_model.pth")}" '
        f'--label-map "{os.path.join(MODELS_DIR, "label_map.json")}" '
        f'--output-dir "{OUTPUT_TEST_DIR}"'
    )
    
    result = os.system(cmd)
    
    if result == 0:
        print("✓ Pipeline test completed successfully!")
        print(f"Check output in: {OUTPUT_TEST_DIR}")
        
        # Try to show summary
        session_dirs = [d for d in os.listdir(OUTPUT_TEST_DIR) if d.startswith('session_')]
        if session_dirs:
            latest_session = sorted(session_dirs)[-1]
            summary_path = os.path.join(OUTPUT_TEST_DIR, latest_session, 'summary.txt')
            report_path = os.path.join(OUTPUT_TEST_DIR, latest_session, 'evaluation_report.html')
            
            if os.path.exists(summary_path):
                print("\n--- Quick Summary ---")
                with open(summary_path, 'r') as f:
                    for line in f.readlines()[:10]:  # Show first 10 lines
                        print(line.rstrip())
                print("...")
            
            if os.path.exists(report_path):
                print(f"\nFull report available at: {report_path}")
        
        return True
    else:
        print("❌ Pipeline test failed!")
        return False

def main():
    print("=" * 60)
    print("Kanji Evaluation System Test")
    print(f"Mode: {'MANUAL FILES' if USE_MANUAL_FILES else 'AUTOMATIC GENERATION'}")
    print("=" * 60)
    
    # First check dependencies
    if not check_dependencies():
        print("\n❌ Please install missing dependencies first!")
        print("\nFor Python packages: pip install -r requirements.txt")
        print("For system packages: sudo apt-get install poppler-utils")
        sys.exit(1)
    
    tests = [
        ("Model Loading", test_model_loading),
        ("PDF Extraction", test_pdf_extraction),
        ("Complete Pipeline", test_complete_pipeline)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    for test_name, success in results:
        status = "✓ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print("\n✓ All tests passed! System is ready to use.")
        print("\nTo switch between manual and automatic mode, edit the file and change:")
        print("  USE_MANUAL_FILES = True/False")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
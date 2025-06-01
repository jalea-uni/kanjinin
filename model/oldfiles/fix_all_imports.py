#!/usr/bin/env python3
"""
Fix all import issues in the inference directory
Run this from the model directory
"""

import os

def fix_evaluate_student_kanji():
    """Fix imports in evaluate_student_kanji.py"""
    file_path = 'inference/evaluate_student_kanji.py'
    
    print(f"Fixing {file_path}...")
    
    # Read the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Find and replace the import lines
    new_lines = []
    for line in lines:
        if 'from model.inference.pdf_box_extractor import PDFBoxExtractor' in line:
            new_lines.append('from pdf_box_extractor import PDFBoxExtractor\n')
        elif 'from model.inference.kanji_evaluator import KanjiEvaluator' in line:
            new_lines.append('from kanji_evaluator import KanjiEvaluator, generate_html_report\n')
        elif 'from .pdf_box_extractor import PDFBoxExtractor' in line:
            new_lines.append('from pdf_box_extractor import PDFBoxExtractor\n')
        elif 'from .kanji_evaluator import KanjiEvaluator' in line:
            new_lines.append('from kanji_evaluator import KanjiEvaluator, generate_html_report\n')
        else:
            new_lines.append(line)
    
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"✓ Fixed {file_path}")

def fix_test_system_paths():
    """Fix the paths in test_system.py to match your structure"""
    file_path = 'inference/test_system.py'
    
    print(f"Fixing {file_path}...")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Update the paths to match your actual structure
    # Replace the path definitions at the top
    new_paths = '''# Directory base del file corrente
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(BASE_DIR, '..'))
MODELS_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'models'))
DATA_TEST_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'data', 'test'))
OUTPUT_TEST_DIR = os.path.abspath(os.path.join(MODEL_DIR, 'output', 'test'))'''
    
    # Find and replace the path section
    import re
    content = re.sub(
        r'# Directory base del file corrente.*?OUTPUT_TEST_DIR = .*?\n',
        new_paths + '\n',
        content,
        flags=re.DOTALL
    )
    
    # Also update the model path references
    content = content.replace(
        'os.path.join(MODELS_DIR, \'best_model.pth\')',
        'os.path.join(MODELS_DIR, \'best_model.pth\')'
    )
    content = content.replace(
        'os.path.join(MODELS_DIR, \'label_map.json\')',
        'os.path.join(MODELS_DIR, \'label_map.json\')'
    )
    
    # Update the evaluator script path
    content = content.replace(
        'evaluator_script = os.path.join(BASE_DIR, \'evaluate_student_kanji.py\')',
        'evaluator_script = os.path.join(BASE_DIR, \'evaluate_student_kanji.py\')'
    )
    
    # Fix the command to include proper paths for model and label map
    old_cmd = '''cmd = (
        f"\"{sys.executable}\" \"{evaluator_script}\" "
        f"\"{test_pdf_path}\" --kanji-list \"{test_list_path}\" "
        f"--output-dir \"{OUTPUT_TEST_DIR}\""
    )'''
    
    new_cmd = '''cmd = (
        f"\"{sys.executable}\" \"{evaluator_script}\" "
        f"\"{test_pdf_path}\" --kanji-list \"{test_list_path}\" "
        f"--model-path \"{os.path.join(MODELS_DIR, 'best_model.pth')}\" "
        f"--label-map \"{os.path.join(MODELS_DIR, 'label_map.json')}\" "
        f"--output-dir \"{OUTPUT_TEST_DIR}\""
    )'''
    
    content = content.replace(old_cmd, new_cmd)
    
    with open(file_path, 'w') as f:
        f.write(content)
    
    print(f"✓ Fixed {file_path}")

def create_init_files():
    """Create __init__.py files"""
    dirs = ['inference', 'shared', 'training']
    
    for dir_name in dirs:
        if os.path.exists(dir_name):
            init_path = os.path.join(dir_name, '__init__.py')
            if not os.path.exists(init_path):
                with open(init_path, 'w') as f:
                    f.write('# Package initialization\n')
                print(f"✓ Created {init_path}")

def create_run_script():
    """Create a convenient run script"""
    script_content = '''#!/bin/bash
# Run test from the correct directory
cd "$(dirname "$0")"
cd inference
python3 test_system.py
'''
    
    with open('run_test.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('run_test.sh', 0o755)
    print("✓ Created run_test.sh")

def main():
    print("Fixing import and path issues...\n")
    
    # Check we're in the right directory
    if not os.path.exists('inference') or not os.path.exists('models'):
        print("Error: Please run this script from the model directory")
        print("Current directory:", os.getcwd())
        return
    
    # Apply fixes
    fix_evaluate_student_kanji()
    fix_test_system_paths()
    create_init_files()
    create_run_script()
    
    print("\n✓ All fixes applied!")
    print("\nYou have two options to run the test:")
    print("1. From model directory: ./run_test.sh")
    print("2. Or manually:")
    print("   cd inference")
    print("   python3 test_system.py")

if __name__ == "__main__":
    main()
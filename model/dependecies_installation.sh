#!/bin/bash

# Setup script for Kanji Evaluation System

echo "Setting up Kanji Evaluation System..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv kanji_eval_env
source kanji_eval_env/bin/activate

# Install requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install additional dependencies for PDF processing
echo "Installing PDF processing dependencies..."
pip install pdf2image
pip install pillow
pip install pypdf2

# Install system dependencies (for pdf2image)
echo ""
echo "NOTE: You may need to install system dependencies:"
echo "  - On Ubuntu/Debian: sudo apt-get install poppler-utils"
echo "  - On macOS: brew install poppler"
echo "  - On Windows: Download poppler and add to PATH"
echo ""

# Download a Japanese font if not provided
echo "Checking for Japanese font..."
if [ ! -f "NotoSansCJK-Regular.ttc" ]; then
    echo "Downloading Noto Sans CJK font..."
    wget https://github.com/googlefonts/noto-cjk/raw/main/Sans/OTC/NotoSansCJK-Regular.ttc
fi

# Create directory structure
echo "Creating directory structure..."
mkdir -p evaluation_output/{extracted,reports,sessions}
mkdir -p test_data

echo ""
echo "Setup complete!"
echo ""
echo "To activate the environment, run: source kanji_eval_env/bin/activate"
echo ""
echo "Example usage:"
echo "  python evaluate_student_kanji.py student_worksheet.pdf --kanji-list kanji_list.json"
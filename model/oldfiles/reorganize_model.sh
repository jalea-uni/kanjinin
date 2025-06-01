#!/bin/bash

# Script to reorganize files within the existing model directory
# Save this as: reorganize_model.sh in the kanjinin/model directory

echo "Reorganizing model directory structure..."

# Check if we're in the right directory
if [ ! -f "train.py" ] || [ ! -f "best_model.pth" ]; then
    echo "Error: Please run this script from the kanjinin/model directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Create subdirectories
mkdir -p training inference shared models evaluation_output/{reports,sessions}

# Move training-related files
echo "Organizing training files..."
mv train.py training/ 2>/dev/null
mv dataset.py training/ 2>/dev/null
mv create_label_map.py training/ 2>/dev/null
mv train.sh training/ 2>/dev/null
mv infer.py training/ 2>/dev/null  # This was part of the original training code

# Move inference/evaluation files
echo "Organizing inference files..."
mv pdf_box_extractor.py inference/ 2>/dev/null
mv kanji_evaluator.py inference/ 2>/dev/null
mv evaluate_student_kanji.py inference/ 2>/dev/null
mv test_system.py inference/ 2>/dev/null

# Move shared files
echo "Organizing shared files..."
mv model.py shared/ 2>/dev/null
mv utils.py shared/ 2>/dev/null

# Move model files
echo "Organizing model files..."
mv best_model.pth models/ 2>/dev/null
mv label_map.json models/ 2>/dev/null

# Create symlinks for shared files
echo "Creating symlinks..."
ln -sf ../shared/model.py training/model.py
ln -sf ../shared/utils.py training/utils.py
ln -sf ../shared/model.py inference/model.py
ln -sf ../shared/utils.py inference/utils.py

# Update imports in inference files to use correct paths
echo "Updating import paths..."

# Fix imports in evaluate_student_kanji.py
if [ -f "inference/evaluate_student_kanji.py" ]; then
    sed -i.bak 's/from pdf_box_extractor/from .pdf_box_extractor/g' inference/evaluate_student_kanji.py
    sed -i.bak 's/from kanji_evaluator/from .kanji_evaluator/g' inference/evaluate_student_kanji.py
    rm inference/evaluate_student_kanji.py.bak
fi

# Create a main evaluation script at the root level
cat > evaluate.py << 'EOF'
#!/usr/bin/env python3
"""
Main entry point for kanji evaluation
Run from kanjinin/model directory
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'inference'))

from inference.evaluate_student_kanji import main

if __name__ == "__main__":
    main()
EOF

chmod +x evaluate.py

# Create convenience bash script
cat > evaluate.sh << 'EOF'
#!/bin/bash
# Convenience script to evaluate a PDF
# Usage: ./evaluate.sh student_worksheet.pdf

if [ $# -eq 0 ]; then
    echo "Usage: ./evaluate.sh <pdf_file>"
    exit 1
fi

# Assuming kanji_list.json is in the parent gen directory
KANJI_LIST="../gen/kanji_list.json"

# Check if kanji_list exists
if [ ! -f "$KANJI_LIST" ]; then
    echo "Warning: kanji_list.json not found at $KANJI_LIST"
    echo "Please specify the correct path with --kanji-list"
    KANJI_LIST="kanji_list.json"
fi

python evaluate.py "$1" \
    --kanji-list "$KANJI_LIST" \
    --model-path models/best_model.pth \
    --label-map models/label_map.json \
    --output-dir evaluation_output
EOF

chmod +x evaluate.sh

# Create README for the new structure
cat > README_structure.md << 'EOF'
# Model Directory Structure

After reorganization:

```
kanjinin/
├── gen/                    # Node.js PDF generation system
│   └── kanji_list.json    # List of kanji to practice
│
└── model/                  # Python ML system
    ├── training/          # Training-related code
    │   ├── train.py
    │   ├── dataset.py
    │   ├── create_label_map.py
    │   └── train.sh
    │
    ├── inference/         # Evaluation system
    │   ├── evaluate_student_kanji.py
    │   ├── pdf_box_extractor.py
    │   ├── kanji_evaluator.py
    │   └── test_system.py
    │
    ├── shared/            # Shared modules
    │   ├── model.py
    │   └── utils.py
    │
    ├── models/            # Trained models
    │   ├── best_model.pth
    │   └── label_map.json
    │
    ├── evaluation_output/ # Results
    │   ├── reports/
    │   └── sessions/
    │
    ├── evaluate.py        # Main entry point
    ├── evaluate.sh        # Convenience script
    └── requirements.txt
```

## Usage

From the `model` directory:
```bash
./evaluate.sh path/to/student_worksheet.pdf
```

Or with Python directly:
```bash
python evaluate.py student_worksheet.pdf --kanji-list ../gen/kanji_list.json
```
EOF

echo "✓ Reorganization complete!"
echo ""
echo "New structure:"
echo "- Training code: training/"
echo "- Inference code: inference/"
echo "- Shared modules: shared/"
echo "- Models: models/"
echo "- Results: evaluation_output/"
echo ""
echo "To evaluate a PDF, run:"
echo "  ./evaluate.sh student_worksheet.pdf"
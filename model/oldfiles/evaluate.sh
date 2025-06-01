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

#!/bin/bash

# evaluate.sh - Simple wrapper for kanji evaluation system
# Usage: ./evaluate.sh test_name [threshold]
# Example: ./evaluate.sh kanjitest2 0.75

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/"

# Default parameters
DEFAULT_THRESHOLD=0.75
DEFAULT_FONT_PATH="$MODEL_DIR/fonts/NotoSansCJK-Regular.ttc"
DATA_DIR="$MODEL_DIR/data"
OUTPUT_DIR="$MODEL_DIR/output"

# Model files
MODEL_PATH="$MODEL_DIR/models/best_model.pth"
LABEL_MAP="$MODEL_DIR/models/label_map.json"

# Python script
EVAL_SCRIPT="$MODEL_DIR/inference/evaluate_student_kanji.py"

# Function to print usage
usage() {
    echo -e "${BLUE}Usage: $0 <test_name> [threshold]${NC}"
    echo
    echo "Arguments:"
    echo "  test_name   - Name of the test (without extension)"
    echo "                Expects: data/test_name.pdf and data/test_name.json"
    echo "  threshold   - SSIM threshold for quality (default: $DEFAULT_THRESHOLD)"
    echo
    echo "Examples:"
    echo "  $0 kanjitest2"
    echo "  $0 kanjitest2 0.8"
    echo "  $0 manual 0.7"
    echo
    echo "The script will look for:"
    echo "  - PDF file: $DATA_DIR/<test_name>.pdf"
    echo "  - JSON file: $DATA_DIR/<test_name>.json"
    exit 1
}

# Function to check if file exists
check_file() {
    local file=$1
    local description=$2
    if [ ! -f "$file" ]; then
        echo -e "${RED}Error: $description not found: $file${NC}"
        return 1
    fi
    echo -e "${GREEN}✓ Found $description${NC}"
    return 0
}

# Check arguments
if [ $# -lt 1 ]; then
    usage
fi

TEST_NAME=$1
THRESHOLD=${2:-$DEFAULT_THRESHOLD}

# Construct file paths
PDF_FILE="$DATA_DIR/${TEST_NAME}.pdf"
JSON_FILE="$DATA_DIR/${TEST_NAME}.json"

# Check for alternative locations (e.g., in data/test/)
if [ ! -f "$PDF_FILE" ] && [ -f "$DATA_DIR/test/${TEST_NAME}.pdf" ]; then
    PDF_FILE="$DATA_DIR/test/${TEST_NAME}.pdf"
fi
if [ ! -f "$JSON_FILE" ] && [ -f "$DATA_DIR/test/${TEST_NAME}.json" ]; then
    JSON_FILE="$DATA_DIR/test/${TEST_NAME}.json"
fi

# Header
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Kanji Evaluation System${NC}"
echo -e "${BLUE}========================================${NC}"
echo
echo -e "Test name: ${YELLOW}$TEST_NAME${NC}"
echo -e "Threshold: ${YELLOW}$THRESHOLD${NC}"
echo

# Check all required files
echo -e "${BLUE}Checking files...${NC}"
all_files_ok=true

check_file "$PDF_FILE" "PDF file" || all_files_ok=false
check_file "$JSON_FILE" "Kanji list JSON" || all_files_ok=false
check_file "$MODEL_PATH" "Model file" || all_files_ok=false
check_file "$LABEL_MAP" "Label map" || all_files_ok=false
check_file "$EVAL_SCRIPT" "Evaluation script" || all_files_ok=false

# Check for font file (optional)
if [ -f "$DEFAULT_FONT_PATH" ]; then
    echo -e "${GREEN}✓ Found font file${NC}"
    FONT_ARG="--font-path $DEFAULT_FONT_PATH"
else
    echo -e "${YELLOW}⚠ Font file not found, using system default${NC}"
    FONT_ARG=""
fi

if [ "$all_files_ok" = false ]; then
    echo
    echo -e "${RED}Missing required files. Please check your installation.${NC}"
    exit 1
fi

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SESSION_DIR="$OUTPUT_DIR/${TEST_NAME}_${TIMESTAMP}"
mkdir -p "$SESSION_DIR"

echo
echo -e "${BLUE}Running evaluation...${NC}"
echo -e "Output directory: ${YELLOW}$SESSION_DIR${NC}"
echo

# Run the evaluation
cd "$MODEL_DIR/inference" || exit 1

python3 "$EVAL_SCRIPT" \
    "$PDF_FILE" \
    --kanji-list "$JSON_FILE" \
    --model-path "$MODEL_PATH" \
    --label-map "$LABEL_MAP" \
    --quality-threshold "$THRESHOLD" \
    --output-dir "$SESSION_DIR" \
    $FONT_ARG

EXIT_CODE=$?

# Check if evaluation was successful
if [ $EXIT_CODE -eq 0 ]; then
    echo
    echo -e "${GREEN}✓ Evaluation completed successfully!${NC}"
    echo
    
    # Show summary if available
    SUMMARY_FILE="$SESSION_DIR/summary.txt"
    if [ -f "$SUMMARY_FILE" ]; then
        echo -e "${BLUE}Summary:${NC}"
        echo "----------------------------------------"
        head -n 15 "$SUMMARY_FILE"
        echo "----------------------------------------"
    fi
    
    # Show report location
    REPORT_FILE=$(find "$SESSION_DIR" -name "evaluation_report.html" | head -1)
    if [ -n "$REPORT_FILE" ]; then
        echo
        echo -e "${GREEN}Full report available at:${NC}"
        echo -e "${YELLOW}$REPORT_FILE${NC}"
        
        # Try to open in browser (optional)
        if command -v xdg-open &> /dev/null; then
            echo
            read -p "Open report in browser? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                xdg-open "$REPORT_FILE"
            fi
        fi
    fi
else
    echo
    echo -e "${RED}✗ Evaluation failed with exit code $EXIT_CODE${NC}"
    echo "Check the error messages above for details."
    exit $EXIT_CODE
fi

echo
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Done!${NC}"
echo -e "${BLUE}========================================${NC}"  
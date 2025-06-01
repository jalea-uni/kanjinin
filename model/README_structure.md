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

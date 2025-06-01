#!/usr/bin/env python3
"""
Complete workflow for evaluating handwritten kanji from student PDFs

Usage:
    python evaluate_student_kanji.py student_worksheet.pdf --kanji-list kanji_list.json
"""

import argparse
import os
import sys
from pathlib import Path
import json
import shutil
from datetime import datetime
from typing import Dict

# Import our modules
from pdf_box_extractor import PDFBoxExtractor
from kanji_evaluator import KanjiEvaluator, generate_html_report


def setup_directories(base_dir: str) -> Dict[str, str]:
    """Create necessary directories for processing"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    dirs = {
        'base': base_dir,
        'extracted': os.path.join(base_dir, 'extracted_boxes'),
        'reports': os.path.join(base_dir, 'reports'),
        'session': os.path.join(base_dir, f'session_{timestamp}')
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate handwritten kanji from student PDFs"
    )
    parser.add_argument('pdf_path', help='Path to student PDF')
    parser.add_argument('--kanji-list', required=True, 
                       help='Path to kanji_list.json with expected characters')
    parser.add_argument('--model-path', default='../models/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--label-map', default='../models/label_map.json',
                       help='Path to label mapping')
    parser.add_argument('--font-path', default=None,
                       help='Path to Japanese font for template rendering')
    parser.add_argument('--output-dir', default='evaluation_output',
                       help='Output directory for results')
    parser.add_argument('--quality-threshold', type=float, default=0.75,
                       help='SSIM threshold for quality assessment')
    parser.add_argument('--min-box-area', type=int, default=5000,
                       help='Minimum area for valid kanji box')
    parser.add_argument('--device', default=None,
                       help='Computing device (cuda/cpu/mps)')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    if not os.path.exists(args.kanji_list):
        print(f"Error: Kanji list not found: {args.kanji_list}")
        sys.exit(1)
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        sys.exit(1)
    
    # Setup directories
    dirs = setup_directories(args.output_dir)
    session_dir = dirs['session']
    
    print(f"Starting evaluation session: {session_dir}")
    
    # Step 1: Extract boxes from PDF
    print("\n1. Extracting kanji boxes from PDF...")
    extractor = PDFBoxExtractor(min_box_area=args.min_box_area)
    
    extraction_results = extractor.process_pdf(
        pdf_path=args.pdf_path,
        output_dir=os.path.join(session_dir, 'boxes'),
        kanji_list_path=args.kanji_list
    )
    
    # Save extraction results
    extraction_results_path = os.path.join(session_dir, 'extraction_results.json')
    with open(extraction_results_path, 'w', encoding='utf-8') as f:
        json.dump(extraction_results, f, ensure_ascii=False, indent=2)
    
    print(f"Extracted {extraction_results['total_boxes']} boxes")
    
    # Step 2: Evaluate extracted kanji
    print("\n2. Evaluating handwritten kanji...")
    evaluator = KanjiEvaluator(
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device,
        quality_threshold=args.quality_threshold
    )
    
    evaluation_results = evaluator.evaluate_batch(
        extraction_results_path=extraction_results_path,
        font_path=args.font_path
    )
    
    # Save evaluation results
    eval_results_path = os.path.join(session_dir, 'evaluation_results.json')
    with open(eval_results_path, 'w', encoding='utf-8') as f:
        json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
    
    # Step 3: Generate report
    print("\n3. Generating evaluation report...")
    report_path = os.path.join(session_dir, 'evaluation_report.html')
    generate_html_report(evaluation_results, report_path)
    
    # Step 4: Create summary file
    summary_path = os.path.join(session_dir, 'summary.txt')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("KANJI EVALUATION SUMMARY\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"PDF File: {os.path.basename(args.pdf_path)}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Kanji Evaluated: {evaluation_results['total_evaluated']}\n")
        f.write(f"Accuracy: {evaluation_results['accuracy']:.1%}\n")
        f.write(f"Quality Rate: {evaluation_results['quality_rate']:.1%}\n\n")
        
        f.write("BREAKDOWN:\n")
        f.write(f"- Perfect (Correct + Good Quality): {evaluation_results['summary']['perfect_count']}\n")
        f.write(f"- Correct but Poor Quality: {evaluation_results['summary']['needs_quality_improvement']}\n")
        f.write(f"- Incorrect: {evaluation_results['summary']['incorrect_count']}\n\n")
        
        f.write("METRICS:\n")
        f.write(f"- Average Confidence: {evaluation_results['summary']['average_confidence']:.1%}\n")
        f.write(f"- Average SSIM Score: {evaluation_results['summary']['average_ssim']:.2f}\n\n")
        
        if evaluation_results['summary']['common_mistakes']:
            f.write("COMMON MISTAKES:\n")
            for expected, predicted in evaluation_results['summary']['common_mistakes']:
                f.write(f"- Expected '{expected}' but got '{predicted}'\n")
    
    # Print final summary
    print("\n" + "=" * 50)
    print("EVALUATION COMPLETE!")
    print("=" * 50)
    print(f"Session Directory: {session_dir}")
    print(f"Report: {report_path}")
    print(f"Accuracy: {evaluation_results['accuracy']:.1%}")
    print(f"Quality Rate: {evaluation_results['quality_rate']:.1%}")
    
    # Copy report to main reports directory with timestamp
    report_filename = f"report_{os.path.basename(args.pdf_path).replace('.pdf', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    shutil.copy(report_path, os.path.join(dirs['reports'], report_filename))
    
    print(f"\nReport also saved to: {os.path.join(dirs['reports'], report_filename)}")
    
    # Create a quick view summary
    print("\n" + "-" * 50)
    print("QUICK SUMMARY:")
    print("-" * 50)
    
    # Show first 5 results
    for i, result in enumerate(evaluation_results['individual_results'][:5]):
        status = "✓" if result['is_correct'] else "✗"
        quality_emoji = "👍" if result['quality']['quality'] == 'good' else "📝"
        print(f"{i+1}. {result['expected']} → {result['predicted']} {status} {quality_emoji}")
    
    if len(evaluation_results['individual_results']) > 5:
        print(f"... and {len(evaluation_results['individual_results']) - 5} more")
    
    print("\nFor detailed results, open the HTML report.")
    
    # Optional: Create CSV export for further analysis
    csv_path = os.path.join(session_dir, 'results.csv')
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write("Index,Expected,Predicted,Correct,Confidence,SSIM,Quality\n")
        for r in evaluation_results['individual_results']:
            f.write(f"{r['box_index']},{r['expected']},{r['predicted']},")
            f.write(f"{r['is_correct']},{r['confidence']:.3f},")
            f.write(f"{r['quality']['ssim_score']:.3f},{r['quality']['quality']}\n")
    
    print(f"\nCSV results saved to: {csv_path}")


if __name__ == "__main__":
    main()
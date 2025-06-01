import torch
import cv2
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
import argparse

# Import from existing modules
from model import get_model
from utils import render_template, compute_ssim
from torchvision import transforms


class KanjiEvaluator:
    def __init__(self, model_path: str, label_map_path: str, 
                 device: str = None, img_size: int = 64,
                 quality_threshold: float = 0.75):
        """
        Initialize Kanji evaluator with trained model
        
        Args:
            model_path: Path to trained model (.pth file)
            label_map_path: Path to label_map.json
            device: Computing device (cuda/cpu/mps)
            img_size: Image size for model input
            quality_threshold: SSIM threshold for quality assessment
        """
        # Auto-detect device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
        
        # Load label map
        with open(label_map_path, 'r') as f:
            self.label_map = json.load(f)
        self.num_classes = len(self.label_map)
        
        # Create reverse mapping: hex_code -> unicode char
        self.code_to_char = {}
        for idx, hex_code in self.label_map.items():
            unicode_val = int(hex_code, 16)
            self.code_to_char[hex_code] = chr(unicode_val)
        
        # Load model
        self.model = get_model(self.num_classes, pretrained=False)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        self.img_size = img_size
        self.quality_threshold = quality_threshold
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image for evaluation"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return img
    
    def predict_kanji(self, image: np.ndarray) -> Tuple[str, int, float]:
        """
        Predict kanji character from image
        
        Returns:
            (predicted_char, class_index, confidence)
        """
        # Transform image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            probs = torch.softmax(output, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        
        # Convert to character
        pred_idx = pred_idx.item()
        confidence = confidence.item()
        hex_code = self.label_map[str(pred_idx)]
        char = self.code_to_char[hex_code]
        
        return char, pred_idx, confidence
    
    def evaluate_quality(self, image: np.ndarray, target_char: str, 
                        font_path: str = None) -> Dict[str, any]:
        """
        Evaluate writing quality using SSIM
        
        Args:
            image: Grayscale image of handwritten kanji
            target_char: Expected kanji character
            font_path: Path to Japanese font (optional)
        
        Returns:
            Dictionary with quality metrics
        """
        # Get unicode value
        unicode_val = ord(target_char)
        
        # Render template
        template = render_template(unicode_val, size=(self.img_size, self.img_size), 
                                 font_path=font_path)
        
        # Resize input image
        img_resized = cv2.resize(image, (self.img_size, self.img_size))
        
        # Compute SSIM
        ssim_score = compute_ssim(img_resized, template)
        
        # Determine quality
        quality = 'good' if ssim_score >= self.quality_threshold else 'needs improvement'
        
        return {
            'ssim_score': float(ssim_score),
            'quality': quality,
            'threshold': self.quality_threshold
        }
    
    def analyze_stroke_order(self, image: np.ndarray, target_char: str) -> Dict[str, any]:
        """
        Basic stroke analysis (can be extended with more sophisticated methods)
        
        Returns:
            Dictionary with stroke analysis
        """
        # This is a placeholder for more advanced stroke analysis
        # Could integrate with stroke order databases or use specialized models
        
        # Simple analysis: check stroke density and distribution
        _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Divide image into quadrants
        h, w = binary.shape
        quadrants = [
            binary[:h//2, :w//2],    # Top-left
            binary[:h//2, w//2:],    # Top-right
            binary[h//2:, :w//2],    # Bottom-left
            binary[h//2:, w//2:]     # Bottom-right
        ]
        
        densities = [np.sum(q > 0) / q.size for q in quadrants]
        
        return {
            'quadrant_densities': densities,
            'total_density': np.sum(binary > 0) / binary.size,
            'feedback': self._generate_stroke_feedback(densities, target_char)
        }
    
    def _generate_stroke_feedback(self, densities: List[float], char: str) -> str:
        """Generate feedback based on stroke distribution"""
        # This is simplified - real implementation would use stroke order data
        if max(densities) - min(densities) > 0.3:
            return "Stroke distribution seems unbalanced. Check the character proportions."
        elif sum(densities) < 0.1:
            return "Strokes appear too thin. Try writing with more confidence."
        elif sum(densities) > 0.4:
            return "Strokes might be too thick. Try using lighter pressure."
        else:
            return "Good stroke balance!"
    
    def evaluate_single(self, image_path: str, expected_kanji: Dict[str, str], 
                       font_path: str = None) -> Dict[str, any]:
        """
        Evaluate a single kanji image
        
        Args:
            image_path: Path to the image
            expected_kanji: Dictionary with kanji info from kanji_list.json
            font_path: Path to font file
        
        Returns:
            Complete evaluation results
        """
        # Load image
        image = self.preprocess_image(image_path)
        
        # Predict kanji
        predicted_char, pred_idx, confidence = self.predict_kanji(image)
        
        # Check if correct
        expected_char = expected_kanji['kanji']
        is_correct = predicted_char == expected_char
        
        # Evaluate quality
        quality_metrics = self.evaluate_quality(image, expected_char, font_path)
        
        # Analyze strokes
        stroke_analysis = self.analyze_stroke_order(image, expected_char)
        
        return {
            'expected': expected_char,
            'predicted': predicted_char,
            'is_correct': is_correct,
            'confidence': float(confidence),
            'quality': quality_metrics,
            'stroke_analysis': stroke_analysis,
            'feedback': self._generate_feedback(is_correct, quality_metrics, confidence)
        }
    
    def _generate_feedback(self, is_correct: bool, quality: Dict, confidence: float) -> str:
        """Generate comprehensive feedback for the student"""
        feedback = []
        
        if not is_correct:
            feedback.append(f"❌ Incorrect character recognized (confidence: {confidence:.2%}).")
            feedback.append("Please review the stroke order and character structure.")
        else:
            feedback.append(f"✓ Correct character! (confidence: {confidence:.2%})")
            
            if quality['quality'] == 'good':
                feedback.append(f"✓ Good writing quality (SSIM: {quality['ssim_score']:.2f})")
            else:
                feedback.append(f"⚠ Writing quality needs improvement (SSIM: {quality['ssim_score']:.2f})")
                feedback.append("Try to match the character proportions more closely.")
        
        return " ".join(feedback)
    
    def evaluate_batch(self, extraction_results_path: str, 
                      font_path: str = None) -> Dict[str, any]:
        """
        Evaluate all extracted boxes from a PDF
        
        Args:
            extraction_results_path: Path to extraction_results.json
            font_path: Path to font file
        
        Returns:
            Complete evaluation report
        """
        # Load extraction results
        with open(extraction_results_path, 'r', encoding='utf-8') as f:
            extraction_data = json.load(f)
        
        results = []
        correct_count = 0
        quality_good_count = 0
        
        for box_info in extraction_data['boxes']:
            if box_info['expected_kanji'] is None:
                continue
            
            eval_result = self.evaluate_single(
                box_info['image_path'],
                box_info['expected_kanji'],
                font_path
            )
            
            eval_result['box_index'] = box_info['index']
            eval_result['page'] = box_info['page']
            
            results.append(eval_result)
            
            if eval_result['is_correct']:
                correct_count += 1
            if eval_result['quality']['quality'] == 'good':
                quality_good_count += 1
        
        total = len(results)
        
        return {
            'total_evaluated': total,
            'correct_count': correct_count,
            'accuracy': correct_count / total if total > 0 else 0,
            'quality_good_count': quality_good_count,
            'quality_rate': quality_good_count / total if total > 0 else 0,
            'individual_results': results,
            'summary': self._generate_summary(results)
        }
    
    def _generate_summary(self, results: List[Dict]) -> Dict[str, any]:
        """Generate summary statistics and recommendations"""
        if not results:
            return {}
        
        # Group by correctness and quality
        groups = {
            'perfect': [],      # Correct + good quality
            'correct_poor': [], # Correct + poor quality
            'incorrect': []     # Incorrect
        }
        
        for r in results:
            if r['is_correct'] and r['quality']['quality'] == 'good':
                groups['perfect'].append(r)
            elif r['is_correct']:
                groups['correct_poor'].append(r)
            else:
                groups['incorrect'].append(r)
        
        # Common mistakes
        incorrect_chars = [(r['expected'], r['predicted']) for r in groups['incorrect']]
        
        return {
            'perfect_count': len(groups['perfect']),
            'needs_quality_improvement': len(groups['correct_poor']),
            'incorrect_count': len(groups['incorrect']),
            'common_mistakes': incorrect_chars[:5],  # Top 5 mistakes
            'average_confidence': np.mean([r['confidence'] for r in results]),
            'average_ssim': np.mean([r['quality']['ssim_score'] for r in results])
        }


# Create evaluation report generator
def generate_html_report(evaluation_results: Dict, output_path: str = "report.html"):
    """Generate an HTML report from evaluation results"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Kanji Practice Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .summary {{ background: #f0f0f0; padding: 20px; border-radius: 10px; }}
            .result {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; }}
            .correct {{ background: #e8f5e9; }}
            .incorrect {{ background: #ffebee; }}
            .score {{ font-size: 24px; font-weight: bold; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        </style>
    </head>
    <body>
        <h1>Kanji Practice Evaluation Report</h1>
        
        <div class="summary">
            <h2>Summary</h2>
            <p class="score">Accuracy: {accuracy:.1%}</p>
            <p>Total Evaluated: {total}</p>
            <p>Correct: {correct} | Quality Good: {quality_good}</p>
            <p>Average Confidence: {avg_conf:.1%}</p>
            <p>Average SSIM Score: {avg_ssim:.2f}</p>
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Index</th>
                <th>Expected</th>
                <th>Predicted</th>
                <th>Correct</th>
                <th>Confidence</th>
                <th>SSIM</th>
                <th>Quality</th>
                <th>Feedback</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """
    
    # Generate table rows
    rows = []
    for r in evaluation_results['individual_results']:
        row_class = 'correct' if r['is_correct'] else 'incorrect'
        row = f"""
        <tr class="{row_class}">
            <td>{r['box_index']}</td>
            <td>{r['expected']}</td>
            <td>{r['predicted']}</td>
            <td>{'✓' if r['is_correct'] else '✗'}</td>
            <td>{r['confidence']:.1%}</td>
            <td>{r['quality']['ssim_score']:.2f}</td>
            <td>{r['quality']['quality']}</td>
            <td>{r['feedback']}</td>
        </tr>
        """
        rows.append(row)
    
    # Fill template
    html = html_template.format(
        accuracy=evaluation_results['accuracy'],
        total=evaluation_results['total_evaluated'],
        correct=evaluation_results['correct_count'],
        quality_good=evaluation_results['quality_good_count'],
        avg_conf=evaluation_results['summary']['average_confidence'],
        avg_ssim=evaluation_results['summary']['average_ssim'],
        rows='\n'.join(rows)
    )
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', default='best_model.pth')
    parser.add_argument('--label-map', default='label_map.json')
    parser.add_argument('--extraction-results', default='extraction_results.json')
    parser.add_argument('--font-path', default=None, help='Path to Japanese font')
    parser.add_argument('--output-report', default='evaluation_report.html')
    parser.add_argument('--device', default=None)
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = KanjiEvaluator(
        model_path=args.model_path,
        label_map_path=args.label_map,
        device=args.device
    )
    
    # Evaluate all boxes
    results = evaluator.evaluate_batch(
        extraction_results_path=args.extraction_results,
        font_path=args.font_path
    )
    
    # Generate report
    generate_html_report(results, args.output_report)
    
    # Print summary
    print(f"\nEvaluation Summary:")
    print(f"Accuracy: {results['accuracy']:.1%}")
    print(f"Quality Rate: {results['quality_rate']:.1%}")
import cv2
import numpy as np
from pdf2image import convert_from_path
import json
from typing import List, Tuple, Dict
import os

class PDFBoxExtractor:
    def __init__(self, min_box_area=5000, border_threshold=150):
        """
        Initialize PDF box extractor
        
        Args:
            min_box_area: Minimum area for a valid box
            border_threshold: Threshold for detecting box borders
        """
        self.min_box_area = min_box_area
        self.border_threshold = border_threshold
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """Convert PDF pages to images"""
        images = convert_from_path(pdf_path, dpi=dpi)
        return [np.array(img) for img in images]
    
    def detect_boxes(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect rectangular boxes in the image
        
        Returns:
            List of (x, y, width, height) tuples
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive threshold to find borders
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # Filter by area and aspect ratio
            if area > self.min_box_area:
                # Check if it's roughly square (kanji boxes are usually square)
                aspect_ratio = w / h
                if 0.7 < aspect_ratio < 1.3:
                    boxes.append((x, y, w, h))
        
        # Sort boxes by position (top to bottom, left to right)
        boxes = sorted(boxes, key=lambda b: (b[1] // 100, b[0]))
        
        return boxes
    
    def extract_box_content(self, image: np.ndarray, box: Tuple[int, int, int, int], 
                          padding: int = 10) -> np.ndarray:
        """
        Extract content from a box with padding removed
        
        Args:
            image: Source image
            box: (x, y, width, height) tuple
            padding: Pixels to remove from borders
        
        Returns:
            Extracted image region
        """
        x, y, w, h = box
        # Add safety margins
        x_start = max(0, x + padding)
        y_start = max(0, y + padding)
        x_end = min(image.shape[1], x + w - padding)
        y_end = min(image.shape[0], y + h - padding)
        
        return image[y_start:y_end, x_start:x_end]
    
    def is_box_empty(self, box_image: np.ndarray, threshold: float = 0.95) -> bool:
        """
        Check if a box is empty (mostly white)
        
        Args:
            box_image: Image of the box content
            threshold: Percentage of white pixels to consider empty
        
        Returns:
            True if box is empty
        """
        gray = cv2.cvtColor(box_image, cv2.COLOR_RGB2GRAY) if box_image.ndim == 3 else box_image
        _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        white_ratio = np.sum(binary == 255) / binary.size
        return white_ratio > threshold
    
    def process_pdf(self, pdf_path: str, output_dir: str, 
                   kanji_list_path: str) -> Dict[str, any]:
        """
        Process entire PDF and extract kanji boxes
        
        Args:
            pdf_path: Path to PDF file
            output_dir: Directory to save extracted images
            kanji_list_path: Path to kanji_list.json
        
        Returns:
            Dictionary with processing results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load expected kanji list
        with open(kanji_list_path, 'r', encoding='utf-8') as f:
            kanji_list = json.load(f)
        
        # Convert PDF to images
        pages = self.pdf_to_images(pdf_path)
        
        results = []
        box_index = 0
        
        for page_num, page_image in enumerate(pages):
            boxes = self.detect_boxes(page_image)
            
            for box in boxes:
                # Extract box content
                box_content = self.extract_box_content(page_image, box)
                
                # Skip if empty
                if self.is_box_empty(box_content):
                    continue
                
                # Save extracted image
                output_path = os.path.join(output_dir, f'box_{box_index:04d}.png')
                cv2.imwrite(output_path, cv2.cvtColor(box_content, cv2.COLOR_RGB2BGR))
                
                # Map to expected kanji if within range
                expected_kanji = None
                if box_index < len(kanji_list):
                    expected_kanji = kanji_list[box_index]
                
                results.append({
                    'index': box_index,
                    'page': page_num,
                    'box_coords': box,
                    'image_path': output_path,
                    'expected_kanji': expected_kanji
                })
                
                box_index += 1
        
        return {
            'total_boxes': len(results),
            'boxes': results
        }


# Example usage
if __name__ == "__main__":
    extractor = PDFBoxExtractor()
    
    # Process student's PDF
    results = extractor.process_pdf(
        pdf_path="student_kanji_sheet.pdf",
        output_dir="extracted_boxes",
        kanji_list_path="kanji_list.json"
    )
    
    print(f"Extracted {results['total_boxes']} boxes")
    
    # Save extraction results
    with open('extraction_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
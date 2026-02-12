from paddleocr import PaddleOCR
from pix2tex.cli import LatexOCR
from PIL import Image
import numpy as np

class OCREngine:
    def __init__(self, lang='korean'):
        # Initialize PaddleOCR
        # use_textline_orientation=False to avoid "list index out of range" error
        # enable_mkldnn=False to avoid "ConvertPirAttribute2RuntimeAttribute" error
        self.ocr = PaddleOCR(use_textline_orientation=False, lang=lang, enable_mkldnn=False)

    def extract_text(self, image):
        """
        Extracts text from an image using PaddleOCR.
        Args:
            image (PIL.Image or np.ndarray): Input image.
        Returns:
            str: Combined extracted text.
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        result = self.ocr.ocr(image)
        
        extracted_text = []
        if result and isinstance(result, list) and len(result) > 0:
            res_item = result[0]
            if isinstance(res_item, dict):
                 if 'rec_texts' in res_item:
                     extracted_text = res_item['rec_texts']
            elif isinstance(res_item, list):
                for line in res_item:
                    text = line[1][0]
                    extracted_text.append(text)
        
        return "\n".join(extracted_text)

    def detect_boxes(self, image):
        """
        Detects text boxes in the image.
        Args:
            image (np.ndarray): Input image.
        Returns:
            list: List of boxes, where each box is a list of points.
        """
        if isinstance(image, Image.Image):
            if image.mode != 'RGB':
                image = image.convert('RGB')
            image = np.array(image)
        elif isinstance(image, np.ndarray):
             if image.ndim == 3 and image.shape[2] == 4:
                 image = Image.fromarray(image).convert('RGB')
                 image = np.array(image)
        
        # Run full pipeline (det + rec) as partial execution is not supported in this version
        # result structure is typically [{'dt_polys': [box_array, ...], 'rec_texts': [...], ...}]
        try:
            result = self.ocr.ocr(image)
        except Exception as e:
            print(f"Error in detect_boxes: {e}")
            return []

        if result and isinstance(result, list) and len(result) > 0:
            # result[0] is likely the dict for the first image
            res_item = result[0]
            if isinstance(res_item, dict):
                # New format: dict with keys 'dt_polys', 'rec_texts', etc.
                if 'dt_polys' in res_item:
                    # dt_polys is a list of numpy arrays
                    return res_item['dt_polys']
            elif isinstance(res_item, list):
                # Old format: [[box, (text, score)]]
                # Filter out None/empty
                return [line[0] for line in res_item if line]
            
        return []

    def recognize_text(self, image_crop):
         """
         Recognize text from a cropped image.
         """
         # Ensure RGB
         if isinstance(image_crop, np.ndarray):
             if image_crop.ndim == 3 and image_crop.shape[2] == 4:
                 image_crop = np.array(Image.fromarray(image_crop).convert('RGB'))
         
         # Run full pipeline on the crop
         try:
             result = self.ocr.ocr(image_crop)
         except Exception as e:
             # Fallback or log error
             return None, 0.0

         if result and isinstance(result, list) and len(result) > 0:
             res_item = result[0]
             
             texts = []
             scores = []
             
             if isinstance(res_item, dict):
                 # New format
                 if 'rec_texts' in res_item:
                     texts = res_item['rec_texts']
                 if 'rec_scores' in res_item:
                     scores = res_item['rec_scores']
             elif isinstance(res_item, list):
                 # Old format: [[box, (text, score)]]
                 texts = [line[1][0] for line in res_item if line]
                 scores = [line[1][1] for line in res_item if line]
             
             if texts:
                 combined_text = " ".join(texts)
                 avg_score = sum(scores) / len(scores) if scores else 0.0
                 return combined_text, avg_score
             
         return None, 0.0
             
         return None, 0.0

class MathEngine:
    def __init__(self):
        # Initialize LatexOCR (Pix2Tex)
        self.model = LatexOCR()

    def extract_latex(self, image):
        """
        Extracts LaTeX formula from an image.
        Args:
            image (PIL.Image): Input image (should be cropped formula).
        Returns:
            str: LaTeX string.
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
            
        try:
            # Pix2Tex expects a PIL Image
            latex_code = self.model(image)
            return latex_code
        except Exception as e:
            return f"Error: {e}"

class MixedEngine:
    def __init__(self):
        self.ocr_engine = OCREngine()
        self.math_engine = MathEngine()

    def extract_mixed(self, image):
        """
        Extracts both Korean text and combined Math formulas.
        Returns a list of segments: [{'type': 'text'|'latex', 'content': ...}]
        """
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            image = Image.fromarray(image)

        # 1. Detect regions using PaddleOCR
        boxes = self.ocr_engine.detect_boxes(image_np)
        
        # Sort boxes top-to-bottom
        # boxes format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        # Sort by y1 (top)
        boxes.sort(key=lambda x: x[0][1])

        segments = []

        for box in boxes:
            # Crop the region
            # Get bounding box coordinates
            xs = [point[0] for point in box]
            ys = [point[1] for point in box]
            left = int(min(xs))
            top = int(min(ys))
            right = int(max(xs))
            bottom = int(max(ys))
            
            # Add padding
            padding = 2
            left = max(0, left - padding)
            top = max(0, top - padding)
            right = min(image.width, right + padding)
            bottom = min(image.height, bottom + padding)

            crop = image_np[top:bottom, left:right]
            
            # 2. Try Text Recognition (Paddle)
            text_res, conf = self.ocr_engine.recognize_text(crop)
            
            # 3. Try Math Recognition (Pix2Tex)
            # Pix2Tex needs PIL Image
            crop_pil = Image.fromarray(crop)
            latex_res = self.math_engine.extract_latex(crop_pil)

            # 4. Heuristics to decide Text vs Math
            is_text = False
            
            # Heuristic A: Korean characters detected -> Text
            # Check for Hangul unicode range
            has_hangul = False
            if text_res:
                for char in text_res:
                    if 0xAC00 <= ord(char) <= 0xD7A3:
                        has_hangul = True
                        break
            
            if has_hangul:
                is_text = True
            
            # Heuristic B: High confidence text and no latex symbols -> Text
            elif conf > 0.95 and "\\" not in latex_res:
                 is_text = True

            # Heuristic C: Latex looks like garbage (very short) -> Text
            # BUT: If it's a valid math symbol like ?, [, ], =, +, -, allow it as Latex
            elif len(latex_res) < 3 and text_res:
                 # valid math symbols that might be short
                 if latex_res in ["?", "[", "]", "(", ")", "+", "-", "=", "*", "/", "√"]:
                     is_text = False
                 else:
                     is_text = True
            
            # Fallback: If text is empty/None but latex has something, default to latex
            if not text_res and latex_res:
                is_text = False
                 
            # Otherwise -> Math
            
            if is_text:
                if text_res:
                    segments.append({'type': 'text', 'content': text_res})
            else:
                 segments.append({'type': 'latex', 'content': f"$$ {latex_res} $$"})
        
        return segments

if __name__ == "__main__":
    # verification code
    print("Engines initialized successfully.")

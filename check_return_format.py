from paddleocr import PaddleOCR
import numpy as np
from PIL import Image

# Create a dummy image (RGBA)
image = np.full((100, 200, 4), 255, dtype=np.uint8)
# Draw a crude "line"
image[40:60, 50:150] = 0

try:
    print("Initializing PaddleOCR with mkldnn=False...")
    ocr = PaddleOCR(use_textline_orientation=False, lang='korean', enable_mkldnn=False)
    
    # Simulate the fix: Convert RGBA to RGB
    if image.ndim == 3 and image.shape[2] == 4:
        image = Image.fromarray(image).convert('RGB')
        image = np.array(image)
        
    print("Running OCR on dummy image (converted to RGB)...")
    # Call without arguments as we know flags are not supported
    result = ocr.ocr(image)
    print("OCR Finished.")
    print("Result Type:", type(result))
    print("Result Content:", result)
    
except Exception as e:
    print(f"Error: {e}")

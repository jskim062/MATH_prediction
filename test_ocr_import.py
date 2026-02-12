try:
    from paddleocr import PaddleOCR
    print("PaddleOCR imported successfully")
    # use_textline_orientation=True replaces use_angle_cls
    ocr = PaddleOCR(use_textline_orientation=True, lang='korean')
    print("PaddleOCR initialized successfully")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")

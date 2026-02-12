import sys
import time
import os

# Set environment variable to skip model source check
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

print("Starting initialization test with PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True...", flush=True)

try:
    from ocr_engine import OCREngine, MathEngine
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

try:
    print("Initializing OCREngine (PaddleOCR)...", flush=True)
    start_time = time.time()
    ocr = OCREngine()
    print(f"OCREngine initialized in {time.time() - start_time:.2f} seconds.", flush=True)
except Exception as e:
    print(f"Failed to initialize OCREngine: {e}", flush=True)

try:
    print("Initializing MathEngine (LatexOCR)...", flush=True)
    start_time = time.time()
    math = MathEngine()
    print(f"MathEngine initialized in {time.time() - start_time:.2f} seconds.", flush=True)
except Exception as e:
    print(f"Failed to initialize MathEngine: {e}", flush=True)

print("Test complete.", flush=True)

# Math & Korean OCR Tool 📝

A Streamlit-based application to extract Korean text and mathematical formulas from images and PDFs using PaddleOCR and Pix2Tex.

## Features
- **Mixed Mode Detection**: Automatically distinguishes between Korean text and mathematical formulas.
- **LaTeX Support**: Converts mathematical formulas into LaTeX format ($$...$$).
- **Multi-format Support**: Upload PNG, JPG, JPEG, or PDF files.
- **Copy-Friendly**: Provides a raw Markdown output for easy copying.
- **Stability Fixes**: Includes specific configurations to handle oneDNN/MKLDNN backend crashes and RGBA image processing.

## Tech Stack
- **PaddleOCR**: High-performance OCR for Korean text detection and recognition.
- **Pix2Tex (LaTeX-OCR)**: Vision-based LaTeX formula extraction.
- **Streamlit**: Fast and interactive web interface.
- **PIL (Pillow)**: Image processing and RGBA to RGB conversion.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jskim062/MATH_prediction.git
   cd MATH_prediction
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements_compatible.txt
   ```

## Usage
Run the application using Streamlit:
```bash
streamlit run app.py
```

## Known Issues & Fixes
- **Backend Crash**: Disabled MKLDNN (`enable_mkldnn=False`) to avoid backend initialization errors on certain environments.
- **RGBA Handling**: Integrated automatic conversion from 4-channel (RGBA) to 3-channel (RGB) to prevent PaddleOCR internal errors.
- **Symbol Recognition**: Optimized heuristics to preserve short symbols like `?`, `[]`, and `√`.

## Future Plans
- [ ] Integration with VLM (Vision Language Model) for enhanced accuracy.
- [ ] Batch processing for large PDF documents.

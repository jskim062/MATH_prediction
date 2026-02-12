# Set environment variable to avoid PaddleOCR network check hang
import os
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import streamlit as st
import time
from PIL import Image
from ocr_engine import MixedEngine
from pdf2image import convert_from_bytes
import numpy as np
import io

# Page Config
st.set_page_config(page_title="Math & Korean OCR", page_icon="📝", layout="wide")

# Initialize Engines (Cached to avoid reloading)
@st.cache_resource
def load_engines():
    return MixedEngine()

mixed_engine = load_engines()

st.title("📝 Math & Korean OCR Tool")
st.markdown("""
Upload a **PDF** or **Image** to extract text.
- **Mixed Mode (Default)**: Automatically detects and extracts both Korean text and Math formulas.
""")

uploaded_file = st.file_uploader("Upload File", type=["png", "jpg", "jpeg", "pdf"])

if uploaded_file:
    # Handle PDF
    if uploaded_file.type == "application/pdf":
        start_time = time.time()
        images = convert_from_bytes(uploaded_file.read())
        process_time = time.time() - start_time
        st.info(f"PDF Loaded in {process_time:.2f} seconds ({len(images)} pages).")
        
        # Select page to view
        page_num = st.number_input("Select Page", min_value=1, max_value=len(images), value=1)
        image = images[page_num - 1]
    else:
        # Handle Images
        start_time = time.time()
        image = Image.open(uploaded_file)
        process_time = time.time() - start_time
        st.success(f"Image Loaded in {process_time:.2f} seconds.")
    
    # Display Image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Process Button
    if st.button("Extract"):
        with st.spinner("Processing..."):
            segments = mixed_engine.extract_mixed(image)
            
            st.subheader("Extracted Content")
            
            # Combine results for display
            full_markdown = ""
            for seg in segments:
                if seg['type'] == 'text':
                    full_markdown += seg['content'] + "\n\n"
                elif seg['type'] == 'latex':
                    full_markdown += seg['content'] + "\n\n"
            
            # Render
            st.markdown(full_markdown)

            st.subheader("Copy to Clipboard")
            st.code(full_markdown, language="markdown")
            
            # Show raw segments for debugging/copying
            with st.expander("Show Raw Segments"):
                st.json(segments)


import streamlit as st
import cv2
import numpy as np
from image_processor import ScoreImageProcessor
from model import get_model, detect_notes
from utils import format_detection_output, get_note_pitch
import io
from PIL import Image

# Page config
st.set_page_config(page_title="Music Score Recognition",
                   page_icon="🎼",
                   layout="wide")

# Load custom CSS
with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Initialize processor and model
@st.cache_resource
def initialize_system():
    processor = ScoreImageProcessor()
    model = get_model()
    return processor, model


processor, model = initialize_system()

# Title
st.title("🎼 Music Score Recognition System")
st.markdown(
    "Upload a sheet music image to detect and analyze musical elements.")

# File upload
with st.container():
    st.markdown('<div class="upload-container">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose a sheet music image (PNG format)",
                                     type=['png'])
    st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    # Convert uploaded file to image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Display original image
    st.image(image, caption="Original Score", use_column_width=True)

    # Processing pipeline with progress bar
    with st.spinner("Processing score..."):
        progress_bar = st.progress(0)

        # 1. Preprocess image
        preprocessed_img, detection_lines = processor.detect_lines(image)
        progress_bar.progress(25)

        progress_bar.progress(100)

    # Display results
    st.markdown('<div class="results-container">', unsafe_allow_html=True)
    st.subheader("Detection Results")

    # Show processed image with detections
    # (In a real implementation, draw detected elements on image)
    st.image(preprocessed_img,
             caption="Processed Score",
             use_column_width=True)
    st.text(f"Detected lines: {detection_lines}")

    # Clear progress bar
    progress_bar.empty()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for music education")

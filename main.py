import streamlit as st
import cv2
import numpy as np
from image_processor import ScoreImageProcessor
from model import get_model
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
    model = get_model(conf_threshold=0.25)  # Configurable confidence threshold
    return processor, model

processor, model = initialize_system()

# Title
st.title("🎼 Music Score Recognition System")
st.markdown(
    "Upload a sheet music image to detect and analyze musical elements.")

# Sidebar for model configuration
with st.sidebar:
    st.header("Model Configuration")
    confidence_threshold = st.slider(
        "Detection Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05
    )

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

        # 1. Detect staff lines
        preprocessed_img, detection_lines = processor.detect_lines(image.copy())
        progress_bar.progress(25)

        # 2. Detect notes using YOLOv5
        notes = model.detect_notes(image)
        progress_bar.progress(50)

        # 3. Draw detections on image
        result_image = image.copy()
        for note in notes:
            x1, y1, x2, y2 = note["bbox"]
            conf = note["confidence"]
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(result_image, f"{conf:.2f}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        progress_bar.progress(100)

        # Display results
        st.markdown('<div class="results-container">', unsafe_allow_html=True)
        st.subheader("Detection Results")

        # Show processed image with detections
        st.image(result_image,
                 caption="Processed Score with Detections",
                 use_column_width=True)

        # Display detection information
        st.write("Staff Lines:", len(detection_lines))
        st.write("Detected Notes:", len(notes))

        # Display JSON output
        if st.checkbox("Show JSON Output"):
            st.json(format_detection_output(detection_lines, notes))

        # Clear progress bar
        progress_bar.empty()

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for music education")

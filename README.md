# Music Score Recognition System

## Overview
A web-based system that analyzes sheet music images to detect and recognize musical elements using computer vision and deep learning.

## Features
- Upload and process sheet music images (PNG format)
- Detect staff lines and musical notes
- Analyze note pitch and duration
- JSON output of detected musical elements
- Real-time processing feedback
- Musical manuscript-inspired interface

## Technology Stack
- Streamlit for web interface
- PyTorch for note detection
- OpenCV for image processing
- NumPy for numerical operations

## Installation
```bash
# Clone the repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt
```

## Usage
1. Run the application:
```bash
streamlit run main.py
```
2. Open browser and navigate to http://localhost:5000
3. Upload a sheet music image (PNG format)
4. View the detection results and JSON output

## Project Structure
- `main.py`: Main Streamlit application
- `image_processor.py`: Image preprocessing and staff line detection
- `model.py`: PyTorch model for note detection
- `utils.py`: Utility functions for data formatting
- `assets/`: UI assets and styling

## Requirements
- Python 3.8+
- PyTorch
- Streamlit
- OpenCV
- NumPy
- SciPy

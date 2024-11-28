import cv2
import numpy as np
from scipy import signal

class ScoreImageProcessor:
    def __init__(self):
        self.staff_lines = []
        
    def preprocess(self, image):
        """Preprocess the image for staff line detection."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        return thresh
    
    def detect_staff_lines(self, preprocessed_image):
        """Detect staff lines using horizontal projection."""
        # Get horizontal projection
        projection = np.sum(preprocessed_image, axis=1)
        
        # Find peaks in projection (staff lines)
        peaks, _ = signal.find_peaks(projection, 
                                   height=preprocessed_image.shape[1]*0.5,
                                   distance=20)
        
        self.staff_lines = sorted(peaks)
        return self.staff_lines
    
    def prepare_for_note_detection(self, image):
        """Prepare image for note detection."""
        # Convert to RGB if needed
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
        # Normalize
        image = image.astype(np.float32) / 255.0
        
        return image

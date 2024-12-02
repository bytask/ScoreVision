import cv2
import numpy as np
from scipy import signal


class ScoreImageProcessor:

    def __init__(self):
        self.staff_lines = []

    # input image and edge detection

    def detect_lines(self, image):
        """Detect lines using Hough Transform in the input image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Perform edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Detect lines using Hough Transform
        # threshold can be adjusted to detect more or less lines
        lines = cv2.HoughLinesP(edges,
                                1,
                                np.pi / 180,
                                threshold=400,
                                minLineLength=100,
                                maxLineGap=10)

        # Store the y-coordinates of the detected lines without duplicates
        y_coordinates = set()
        if lines is not None:
            for line in lines:
                for x1, y1, x2, y2 in line:
                    y_coordinates.add(y1)
                    # Draw the detected line on the original image
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return image, list(y_coordinates)

    def edge_detection(self, image):
        """Perform edge detection on the input image."""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Perform Canny edge detection
        edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

        return edges

    # def preprocess(self, image):
    #     """Preprocess the image for staff line detection."""
    #     # Convert to grayscale
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #     # Apply adaptive thresholding
    #     thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #                                    cv2.THRESH_BINARY_INV, 15, 10)

    #     return thresh

    def detect_staff_lines(self, preprocessed_image):
        """Detect staff lines using horizontal projection."""
        # Get horizontal projection
        projection = np.sum(preprocessed_image, axis=1)

        # Find peaks in projection (staff lines)
        peaks, _ = signal.find_peaks(projection,
                                     height=preprocessed_image.shape[1] * 0.5,
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

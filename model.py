from ultralytics import YOLO
import torch
import numpy as np

class NoteDetectionModel:
    def __init__(self, conf_threshold=0.25):
        """Initialize YOLOv5 model for music note detection."""
        self.model = YOLO('yolov5s.pt')  # Load the default YOLOv5 small model
        self.conf_threshold = conf_threshold
        
    def preprocess_image(self, image):
        """Preprocess image for YOLOv5 inference."""
        # YOLOv5 expects BGR images
        if len(image.shape) == 2:  # If grayscale
            image = np.stack((image,) * 3, axis=-1)
        return image

    def detect_notes(self, image):
        """
        Detect musical notes in the image using YOLOv5.
        
        Args:
            image: numpy array of shape (H, W, C)
            
        Returns:
            List of dictionaries containing detected notes information
        """
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Run inference
        results = self.model(processed_image, conf=self.conf_threshold)
        
        # Process results
        notes = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                
                # Create note dictionary
                note = {
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf,
                    "class": cls,
                    "duration": "quarter"  # Default duration, to be refined based on detection
                }
                notes.append(note)
        
        return notes

def get_model(conf_threshold=0.25):
    """Initialize and return the note detection model."""
    return NoteDetectionModel(conf_threshold=conf_threshold)

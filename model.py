import torch
import torch.nn as nn
import torchvision.models as models

class NoteDetectionModel(nn.Module):
    def __init__(self, num_classes=2):  # 2 classes: note head and rest
        super(NoteDetectionModel, self).__init__()
        
        # Use ResNet18 as backbone
        self.backbone = models.resnet18(pretrained=True)
        
        # Modify for our use case
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes * 4)  # 4 values for bbox
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model():
    """Initialize and return the note detection model."""
    model = NoteDetectionModel()
    model.eval()
    return model

def detect_notes(model, image):
    """Detect notes in the preprocessed image."""
    # Convert image to tensor
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        predictions = model(image_tensor)
    
    # Process predictions (simplified)
    notes = []
    pred = predictions[0].numpy()
    
    # Simulate some detected notes for demonstration
    # In a real implementation, this would process actual model outputs
    notes = [
        {"bbox": [100, 150, 120, 170], "duration": "quarter"},
        {"bbox": [200, 130, 220, 150], "duration": "half"},
        {"bbox": [300, 160, 320, 180], "duration": "quarter"}
    ]
    
    return notes

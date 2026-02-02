import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class EfficientNetB4Native(nn.Module):
    """
    EfficientNet-B4 model adapted for the Animal Classification task.

    Key Features:
    - Backbone: EfficientNet-B4 (pretrained on ImageNet).
    - Pooling: Hybrid pooling (concatenation of Global Average Pooling and Global Max Pooling)
      to capture both general scene context and peak features of small objects.
    - Resolution: Designed for 380x380 inputs (handled by dataset transforms).
    """

    def __init__(self):
        super(EfficientNetB4Native, self).__init__()

        # Load weights based on Config
        if Config.PRETRAINED:
            weights = models.EfficientNet_B4_Weights.DEFAULT
        else:
            weights = None

        # Load the full model
        original_model = models.efficientnet_b4(weights=weights)

        # Isolate the feature extractor (convolutional layers)
        # This removes the original AvgPool and Classifier
        self.backbone = original_model.features

        # Determine the output feature dimension dynamically
        # This ensures compatibility even if the backbone changes slightly
        with torch.no_grad():
            # Use a dummy input to get feature shape.
            # Size doesn't strictly matter for channel count, but 380 is native.
            dummy_input = torch.zeros(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # Define the custom classifier
        # We concatenate GAP and GMP, so the input dimension is num_features * 2
        self.classifier = nn.Linear(self.num_features * 2, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images (B, 3, H, W).

        Returns:
            torch.Tensor: Class logits (B, NUM_CLASSES).
        """
        # 1. Backbone Feature Extraction
        # Output shape: (Batch_Size, Channels, Height_Feature, Width_Feature)
        x = self.backbone(x)

        # 2. Hybrid Pooling
        # Global Average Pooling: Captures overall scene context
        avg_pool = torch.mean(x, dim=(2, 3))

        # Global Max Pooling: Captures most prominent features (good for small animals)
        max_pool = torch.amax(x, dim=(2, 3))

        # Concatenate the pooling outputs
        # Shape: (Batch_Size, Channels * 2)
        x = torch.cat((avg_pool, max_pool), dim=1)

        # 3. Classification Head
        # Shape: (Batch_Size, NUM_CLASSES)
        logits = self.classifier(x)

        return logits

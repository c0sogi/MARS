import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


class EfficientNetClassifier(nn.Module):
    """
        EfficientNet-B4 based classifier with Concatenated Global Pooling.

        This model uses a pretrained EfficientNet-B4 backbone to extract features.
        It replaces the standard classifier with a custom head that concatenates
    >>>>>>> REPLACE
    <<<<<<< SEARCH
            # Load Pretrained Backbone
            # We use the 'DEFAULT' weights which correspond to the best available weights for the model
            weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b3(weights=weights)

            # Extract the feature extractor (all layers except the classifier and avgpool)
    =======
            # Load Pretrained Backbone
            # We use the 'DEFAULT' weights which correspond to the best available weights for the model
            weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
            base_model = models.efficientnet_b4(weights=weights)

            # Extract the feature extractor (all layers except the classifier and avgpool)
        Global Average Pooling (GAP) and Global Max Pooling (GMP) outputs,
        followed by a linear classification layer.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Initialize the EfficientNetClassifier.

        Args:
            num_classes (int): The number of output classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to use ImageNet pretrained weights. Defaults to Config.PRETRAINED.
        """
        super(EfficientNetClassifier, self).__init__()

        # Load Pretrained Backbone
        # We use the 'DEFAULT' weights which correspond to the best available weights for the model
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        base_model = models.efficientnet_b3(weights=weights)

        # Extract the feature extractor (all layers except the classifier and avgpool)
        # torchvision's EfficientNet implementation stores the convolutional layers in .features
        self.backbone = base_model.features

        # Dynamically determine the number of input features for the linear layer.
        # This ensures compatibility even if the backbone architecture changes slightly.
        # We perform a dummy forward pass with a zero tensor to get the output shape.
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            dummy_features = self.backbone(dummy_input)
            self.num_features = dummy_features.shape[1]

        # Define Pooling Layers
        # Adaptive pooling allows us to handle variable input sizes if necessary,
        # though we stick to fixed size here. Output is always (B, C, 1, 1).
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))

        # Define Custom Classification Head
        # We concatenate the outputs of Avg and Max pooling, so input dim is num_features * 2
        self.dropout = nn.Dropout(p=Config.DROPOUT_RATE)
        self.fc = nn.Linear(self.num_features * 2, num_classes)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch, Num_Classes).
        """
        # 1. Feature Extraction
        # Pass input through the EfficientNet backbone
        x = self.backbone(x)

        # 2. Global Pooling
        # Apply both Average and Max pooling
        avg_out = self.avg_pool(x).flatten(1)
        max_out = self.max_pool(x).flatten(1)

        # 3. Concatenation
        # Combine the features. This retains both background context (Avg)
        # and peak activation features (Max), useful for detecting small animals.
        x = torch.cat([avg_out, max_out], dim=1)

        # 4. Classification
        x = self.dropout(x)
        logits = self.fc(x)

        return logits

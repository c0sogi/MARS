import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class FractureMILModel(nn.Module):
    """
    Anatomically-Guided Max-Pooling MIL Network.
    Uses ResNet18 backbone and Positional Injection with Global Max Pooling.
    """

    def __init__(self):
        super(FractureMILModel, self).__init__()

        # 1. Backbone: ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Remove the final FC layer.
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        # Feature dimension from ResNet18 is 512
        self.feature_dim = 512

        # 2. Positional Injection
        # We append 1 dimension (normalized depth) to the features
        self.mil_input_dim = self.feature_dim + 1

        # 3. Classifier Head (Instance Level)
        # Maps instance features to 7 classes (C1-C7)
        self.classifier = nn.Linear(self.mil_input_dim, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input volume of shape (Batch, Slices, Channels, Height, Width)
                              Channels should be 3 (2.5D stacking).

        Returns:
            torch.Tensor: Logits for C1-C7 of shape (Batch, 7)
        """
        batch_size, num_slices, channels, height, width = x.shape

        # --- 1. Feature Extraction ---
        # Reshape to process all slices in parallel: (B*S, C, H, W)
        x_flat = x.view(batch_size * num_slices, channels, height, width)

        # Pass through backbone
        features = self.backbone(x_flat)  # (B*S, 512, 1, 1)
        features = features.view(features.size(0), -1)  # Flatten to (B*S, 512)

        # --- 2. Positional Injection ---
        # Generate normalized depth coordinates [0, 1]
        device = x.device
        pos_encoding = torch.linspace(0, 1, steps=num_slices, device=device)
        pos_encoding = pos_encoding.view(1, num_slices, 1).expand(batch_size, -1, -1)
        pos_encoding_flat = pos_encoding.reshape(batch_size * num_slices, 1)

        # Concatenate features and positional encoding
        # (B*S, 512) cat (B*S, 1) -> (B*S, 513)
        features_aug = torch.cat([features, pos_encoding_flat], dim=1)

        # --- 3. Instance Classification ---
        # Predict logits for each slice: (B*S, 7)
        slice_logits = self.classifier(features_aug)

        # --- 4. Sequence Aggregation (Max Pooling) ---
        # Reshape to (B, S, 7)
        slice_logits = slice_logits.view(batch_size, num_slices, -1)

        # Max pool over slices
        # logits: (B, 7)
        logits, _ = torch.max(slice_logits, dim=1)

        return logits

import torch
import torch.nn as nn
import timm
from library.config import Config


class FractureMILModel(nn.Module):
    """
    2D Multiple Instance Learning (MIL) Model for Cervical Spine Fracture Detection.

    This model treats a 3D CT scan as a 'bag' of 2D slices. It processes each slice
    independently using a 2D CNN backbone and aggregates the predictions using
    Max-Pooling to generate a study-level probability for each fracture type.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        """
        Args:
            model_name (str): The name of the timm backbone to use (default: efficientnet_b0).
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(FractureMILModel, self).__init__()

        # Initialize the 2D CNN backbone using timm.
        # - in_chans=1: Adapts the first layer for grayscale CT inputs.
        # - num_classes=Config.N_CLASSES (8): Sets the final linear layer to output
        #   logits for C1-C7 and patient_overall.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=Config.IN_CHANNELS,
            num_classes=Config.N_CLASSES,
        )

    def forward(self, x):
        """
        Forward pass of the MIL model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, Slices, Channels, Height, Width).

        Returns:
            torch.Tensor: Predicted probabilities of shape (Batch, N_CLASSES).
        """
        # Unpack dimensions
        # b: Batch size (number of studies)
        # s: Number of slices per study
        # c: Channels (1)
        # h, w: Image height and width
        b, s, c, h, w = x.shape

        # 1. Collapse Batch and Slice dimensions
        # Reshape to (Batch * Slices, C, H, W) to process all slices as independent images
        x = x.view(b * s, c, h, w)

        # 2. Slice-level Inference
        # Pass through the backbone to get logits for each slice
        # Shape: (Batch * Slices, N_CLASSES)
        logits = self.backbone(x)

        # 3. Reshape back to Study structure
        # Shape: (Batch, Slices, N_CLASSES)
        logits = logits.view(b, s, Config.N_CLASSES)

        # 4. MIL Aggregation (Max Pooling)
        # We take the maximum logit across the slice dimension (dim=1).
        # This selects the most confident prediction for each class within the study.
        # Shape: (Batch, N_CLASSES)
        pooled_logits, _ = torch.max(logits, dim=1)

        # 5. Activation
        # Convert logits to probabilities [0, 1]
        probs = torch.sigmoid(pooled_logits)

        return probs

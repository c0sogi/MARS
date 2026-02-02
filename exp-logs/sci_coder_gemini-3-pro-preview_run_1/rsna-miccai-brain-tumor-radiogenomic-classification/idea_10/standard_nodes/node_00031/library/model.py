import torch
import torch.nn as nn
import timm
from library.config import MODEL_BACKBONE, PRETRAINED, DROP_RATE, NUM_CLASSES
from library.utils import set_seed

# Ensure reproducibility upon module import
set_seed()


class TMSVNet(nn.Module):
    """
    Tri-Stream Modality-Specific Volumetric Network (TMSV-Net).

    This architecture uses three parallel EfficientNet-B0 backbones to independently
    process FLAIR, T1wCE, and T2w MRI slabs. The spatial features are fused via
    concatenation and processed by a classification head.
    """

    def __init__(
        self,
        backbone_name=MODEL_BACKBONE,
        pretrained=PRETRAINED,
        drop_rate=DROP_RATE,
        num_classes=NUM_CLASSES,
    ):
        """
        Args:
            backbone_name (str): Name of the timm backbone (default: efficientnet_b0).
            pretrained (bool): Whether to load ImageNet weights.
            drop_rate (float): Dropout probability for the fusion head.
            num_classes (int): Number of output classes (default: 1).
        """
        super(TMSVNet, self).__init__()

        # Stream 1: FLAIR
        # Input: 3 channels (z-1, z, z+1)
        self.flair_stream = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # Use Global Average Pooling, remove classification head
            in_chans=3,
        )

        # Stream 2: T1wCE
        # Input: 3 channels (z-1, z, z+1)
        self.t1wce_stream = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, in_chans=3
        )

        # Stream 3: T2w
        # Input: 3 channels (z-1, z, z+1)
        self.t2w_stream = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, in_chans=3
        )

        # Determine feature dimension dynamically
        # EfficientNet-B0 typically outputs 1280 features after GAP
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            # All streams are identical in architecture, so check one
            feature_dim = self.flair_stream(dummy_input).shape[1]

        self.fusion_dim = feature_dim * 3

        # Fusion Head
        # Concatenate features -> Dropout -> Dense
        self.dropout = nn.Dropout(p=drop_rate)
        self.classifier = nn.Linear(self.fusion_dim, num_classes)

    def forward(self, x_flair, x_t1wce, x_t2w):
        """
        Forward pass of the TMSV-Net.

        Args:
            x_flair (torch.Tensor): Batch of FLAIR slabs (B, 3, H, W).
            x_t1wce (torch.Tensor): Batch of T1wCE slabs (B, 3, H, W).
            x_t2w (torch.Tensor): Batch of T2w slabs (B, 3, H, W).

        Returns:
            torch.Tensor: Logits (B, num_classes).
        """
        # Extract features from each stream
        # Output shape: (B, feature_dim) e.g., (B, 1280)
        f_flair = self.flair_stream(x_flair)
        f_t1wce = self.t1wce_stream(x_t1wce)
        f_t2w = self.t2w_stream(x_t2w)

        # Feature Fusion
        # Concatenate along the feature dimension
        # Output shape: (B, feature_dim * 3) e.g., (B, 3840)
        f_fused = torch.cat([f_flair, f_t1wce, f_t2w], dim=1)

        # Classification Head
        x = self.dropout(f_fused)
        logits = self.classifier(x)

        return logits

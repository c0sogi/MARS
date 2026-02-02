import torch
import torch.nn as nn
import timm


class SDVNet(nn.Module):
    """
    Siamese Dual-View 2.5D Network (SDV-Net).

    This architecture processes high-density MRI volumes by splitting them into two
    interleaved streams (Even and Odd slices). Both streams are processed by a
    shared EfficientNet-B0 backbone to extract features, which are then fused
    via concatenation and passed to a classification head.

    Attributes:
        backbone (nn.Module): Shared EfficientNet-B0 encoder.
        classifier (nn.Module): Fully connected layer for final prediction.
    """

    def __init__(self, model_name="efficientnet_b0", pretrained=True):
        """
        Initializes the SDVNet model.

        Args:
            model_name (str): Name of the timm model to use as backbone.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(SDVNet, self).__init__()

        # Configuration based on task description:
        # - in_chans=64: 16 slices * 4 modalities per stream.
        # - num_classes=0: Return the global average pooled feature vector.
        # - drop_path_rate=0.2: Regularization for the backbone.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            num_classes=0,
            drop_path_rate=0.2,
        )

        # Determine the output feature dimension of the backbone
        if hasattr(self.backbone, "num_features"):
            self.num_features = self.backbone.num_features
        else:
            # Fallback inspection if num_features attribute is missing
            # Create a dummy input to infer shape
            with torch.no_grad():
                dummy = torch.zeros(1, 64, 256, 256)
                out = self.backbone(dummy)
                self.num_features = out.shape[1]

        # Fusion Head
        # Concatenates features from Even and Odd streams (Size * 2) -> Single Logit
        self.classifier = nn.Linear(self.num_features * 2, 1)

    def forward(self, x_even, x_odd):
        """
        Forward pass of the Siamese network.

        Args:
            x_even (torch.Tensor): Input tensor for Even view slices. Shape (B, 64, H, W).
            x_odd (torch.Tensor): Input tensor for Odd view slices. Shape (B, 64, H, W).

        Returns:
            torch.Tensor: Logits for the target class. Shape (B, 1).
        """
        # Pass both streams through the shared backbone
        # Output shape: (Batch_Size, Num_Features)
        f_even = self.backbone(x_even)
        f_odd = self.backbone(x_odd)

        # Late Fusion: Concatenate feature vectors
        # Output shape: (Batch_Size, Num_Features * 2)
        f_cat = torch.cat([f_even, f_odd], dim=1)

        # Final Classification
        # Output shape: (Batch_Size, 1)
        logits = self.classifier(f_cat)

        return logits

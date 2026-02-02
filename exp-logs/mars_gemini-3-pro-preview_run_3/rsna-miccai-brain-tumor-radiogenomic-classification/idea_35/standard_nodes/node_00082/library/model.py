import torch
import torch.nn as nn
import timm


class DSSVNet(nn.Module):
    """
    Dual-Stream Strided-View 2.5D Network (DSSV-Net).

    This architecture implements a Siamese network strategy where a shared
    EfficientNet-B0 backbone processes two strided views (Even and Odd) of the
    MRI volume. The features are fused via concatenation and passed to a
    linear classifier.
    """

    def __init__(self, model_name: str = "efficientnet_b0", pretrained: bool = True):
        """
        Args:
            model_name (str): Name of the timm model to use. Defaults to 'efficientnet_b0'.
            pretrained (bool): Whether to load pretrained ImageNet weights.
        """
        super(DSSVNet, self).__init__()

        # Initialize the shared backbone using timm
        # in_chans=64: Adapts the first layer to accept 64 channels (16 slices * 4 modalities)
        # num_classes=0: Removes the original classifier and returns the pooled feature vector
        # drop_path_rate=0.2: Applies Stochastic Depth regularization
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=64,
            num_classes=0,
            drop_path_rate=0.2,
        )

        # Retrieve the number of output features from the backbone
        # For EfficientNet-B0, this is typically 1280
        num_features = self.backbone.num_features

        # Fusion Head
        # We concatenate the features from the Even and Odd streams, so the input
        # dimension is num_features * 2. The output is 1 logit for binary classification.
        self.classifier = nn.Linear(num_features * 2, 1)

    def forward(self, x_even: torch.Tensor, x_odd: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the DSSV-Net.

        Args:
            x_even (torch.Tensor): Input tensor for the Even view.
                                   Shape: (Batch, 64, Height, Width)
            x_odd (torch.Tensor): Input tensor for the Odd view.
                                  Shape: (Batch, 64, Height, Width)

        Returns:
            torch.Tensor: Logits for the target class. Shape: (Batch, 1)
        """
        # Pass the Even view through the shared backbone
        # Output shape: (Batch, num_features)
        feat_even = self.backbone(x_even)

        # Pass the Odd view through the shared backbone
        # Output shape: (Batch, num_features)
        feat_odd = self.backbone(x_odd)

        # Late Fusion: Concatenate the feature vectors along the channel dimension
        # Output shape: (Batch, num_features * 2)
        combined_features = torch.cat([feat_even, feat_odd], dim=1)

        # Pass through the classifier head to get the final prediction
        # Output shape: (Batch, 1)
        logits = self.classifier(combined_features)

        return logits

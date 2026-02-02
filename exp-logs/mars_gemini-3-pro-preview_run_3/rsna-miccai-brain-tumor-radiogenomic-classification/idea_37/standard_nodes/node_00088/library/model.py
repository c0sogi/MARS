import torch
import torch.nn as nn
import timm
from library import config


class SiameseSNRNet(nn.Module):
    """
    Siamese Native-Resolution 2.5D Network (SNR-Net).

    This model implements a Siamese architecture where two views of the 3D volume
    (Even slices and Odd slices) are processed by a shared EfficientNet-B0 backbone.
    The resulting feature vectors are concatenated and passed through a linear
    classifier to predict the MGMT promoter methylation status.
    """

    def __init__(self):
        super(SiameseSNRNet, self).__init__()

        # Initialize the shared backbone using timm
        # config.BACKBONE: "efficientnet_b0"
        # config.IN_CHANS: 64 (16 slices * 4 modalities)
        # config.DROP_PATH_RATE: 0.2 (Stochastic Depth)
        # num_classes=0 and global_pool='avg' ensure we get the pooled feature vector
        self.backbone = timm.create_model(
            config.BACKBONE,
            pretrained=True,
            in_chans=config.IN_CHANS,
            num_classes=0,
            global_pool="avg",
            drop_path_rate=config.DROP_PATH_RATE,
        )

        # Retrieve the feature dimension size from the backbone
        # For EfficientNet-B0, this is typically 1280
        self.num_features = self.backbone.num_features

        # Fusion Head
        # We concatenate the features from the Even and Odd streams,
        # so the input dimension is num_features * 2.
        # Output is 1 logit for binary classification.
        self.classifier = nn.Linear(self.num_features * 2, 1)

    def forward(self, x_even, x_odd):
        """
        Forward pass of the Siamese Network.

        Args:
            x_even (torch.Tensor): Input tensor for the Even stream.
                                   Shape: (Batch_Size, 64, 224, 224)
            x_odd (torch.Tensor): Input tensor for the Odd stream.
                                  Shape: (Batch_Size, 64, 224, 224)

        Returns:
            torch.Tensor: Predicted logits. Shape: (Batch_Size, 1)
        """
        # Pass the Even stream through the shared backbone
        # Output Shape: (Batch_Size, num_features)
        feat_even = self.backbone(x_even)

        # Pass the Odd stream through the shared backbone
        # Output Shape: (Batch_Size, num_features)
        feat_odd = self.backbone(x_odd)

        # Late Fusion: Concatenate the feature vectors along the channel dimension
        # Output Shape: (Batch_Size, num_features * 2)
        combined = torch.cat([feat_even, feat_odd], dim=1)

        # Pass through the final classification layer
        # Output Shape: (Batch_Size, 1)
        logits = self.classifier(combined)

        return logits

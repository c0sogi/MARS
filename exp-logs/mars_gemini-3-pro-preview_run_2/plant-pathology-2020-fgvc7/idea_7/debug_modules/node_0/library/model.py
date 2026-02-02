import torch
import torch.nn as nn
import timm
from library.config import Config


class MultiSampleDropout(nn.Module):
    """
    Multi-Sample Dropout module.
    Applies multiple dropout masks to the input features and passes them
    through a shared linear layer. This technique accelerates convergence
    and improves generalization.
    """

    def __init__(self, in_features, out_features, num_dropouts, dropout_rate):
        super().__init__()
        # Multiple dropout layers with different internal states (masks)
        self.dropouts = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(num_dropouts)]
        )
        # Shared linear layer
        self.fc = nn.Linear(in_features, out_features)

        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input features (B, in_features)

        Returns:
            list[torch.Tensor] if training: List of logits from each dropout path.
            torch.Tensor if eval: Logits from the linear layer (dropout disabled).
        """
        if self.training:
            # Apply each dropout mask and then the shared linear layer
            # Returns a list of tensors for loss averaging
            return [self.fc(dropout(x)) for dropout in self.dropouts]
        else:
            # During inference, dropout is identity.
            # This is mathematically equivalent to the average of the branches.
            return self.fc(x)


class AppleClassifier(nn.Module):
    """
    Apple Disease Classifier using a timm backbone and Multi-Sample Dropout head.
    Supports heterogeneous backbones (EfficientNetV2, ConvNeXt) via model_name.
    """

    def __init__(self, model_name, pretrained=True):
        super().__init__()

        # Create backbone using timm
        # num_classes=0 and global_pool='avg' ensures the model outputs
        # a flattened feature vector (B, num_features)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine input features for the head based on backbone architecture
        in_features = self.backbone.num_features

        # Create Multi-Sample Dropout Head
        # Targets: [Rust, Scab] -> 2 output neurons
        self.head = MultiSampleDropout(
            in_features=in_features,
            out_features=len(Config.TARGET_COLS),
            num_dropouts=Config.MSD_NUM_DROPOUTS,
            dropout_rate=Config.MSD_DROPOUT_RATE,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images (B, C, H, W)

        Returns:
            list[torch.Tensor] (Training): List of logits for MSD loss.
            torch.Tensor (Inference): Averaged logits.
        """
        # Extract features from backbone
        features = self.backbone(x)

        # Pass through MSD head
        return self.head(features)

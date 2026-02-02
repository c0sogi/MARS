import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    Bird Species Classification Model.

    Implements a flexible backbone architecture using `timm` with a
    Multi-Sample Dropout head for improved regularization and generalization.
    """

    def __init__(self, backbone_name: str, pretrained: bool = True):
        """
        Args:
            backbone_name (str): Name of the backbone model (e.g., 'resnet18').
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(BirdModel, self).__init__()

        # Create the backbone model
        # num_classes=0 and global_pool='avg' ensures we get the pooled feature vector
        # in_chans=3 handles the Pseudo-RGB input
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            in_chans=3,
        )

        # Determine the number of input features for the classification head
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback for some models where num_features might not be directly exposed
            # Run a dummy forward pass to determine shape
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.in_features = features.shape[1]

        # Multi-Sample Dropout Head
        # We create multiple dropout layers with different random masks
        # The Linear layer is shared across all dropout samples
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.DROPOUT_RATE) for _ in range(Config.NUM_DROPOUT_SAMPLES)]
        )

        self.fc = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, NumClasses).
        """
        # Extract pooled features from the backbone
        # Shape: (Batch, In_Features)
        features = self.backbone(x)

        # Apply Multi-Sample Dropout
        # Pass features through each dropout layer, then the shared FC layer
        logits_list = []
        for dropout_layer in self.dropouts:
            logits_list.append(self.fc(dropout_layer(features)))

        # Stack the results and compute the mean across the dropout samples
        # Shape: (Batch, NumClasses)
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        return logits

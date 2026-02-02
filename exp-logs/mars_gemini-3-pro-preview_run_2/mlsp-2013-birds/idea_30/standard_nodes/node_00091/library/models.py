import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    Heterogeneous Bird Species Classification Model.

    Supports ResNet18, EfficientNet-B0, and DenseNet121 backbones via timm.
    Implements a Multi-Sample Dropout head to improve generalization and stability.
    """

    def __init__(
        self,
        backbone_name: str,
        num_classes: int = Config.NUM_CLASSES,
        pretrained: bool = True,
    ):
        """
        Args:
            backbone_name (str): Name of the backbone architecture (e.g., 'resnet18').
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(BirdModel, self).__init__()
        self.backbone_name = backbone_name

        # Create backbone using timm
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        # global_pool='' ensures we handle pooling if needed, but usually num_classes=0 with
        # default global_pool in timm returns the pooled vector (Batch, Num_Features).
        # We explicitly set global_pool='avg' to be sure.
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Determine the input feature dimension for the classifier
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback for some models if num_features isn't exposed directly
            # Run a dummy forward pass
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy_input)
                self.in_features = features.shape[1]

        # Multi-Sample Dropout Head
        # We use the dropout rates specified in Config
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in Config.DROPOUT_RATES])

        # Shared Fully Connected Layer
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input images of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        # Extract features from backbone
        # Shape: (Batch, In_Features)
        features = self.backbone(x)

        # Multi-Sample Dropout
        # Apply each dropout layer, then the shared FC layer
        logits_list = []
        for dropout_layer in self.dropouts:
            dropped_features = dropout_layer(features)
            logits = self.fc(dropped_features)
            logits_list.append(logits)

        # Stack and average the logits
        # Shape: (Batch, Num_Classes)
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        return logits


def get_model(backbone_name: str, device: torch.device = Config.DEVICE) -> nn.Module:
    """
    Factory function to instantiate and move the model to the specified device.

    Args:
        backbone_name (str): Name of the backbone (must be in Config.BACKBONES).
        device (torch.device): Device to load the model onto.

    Returns:
        nn.Module: The initialized BirdModel.
    """
    model = BirdModel(backbone_name=backbone_name)
    model.to(device)
    return model

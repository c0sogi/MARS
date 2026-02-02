import torch
import torch.nn as nn
import timm
from library.config import Config


class BirdModel(nn.Module):
    """
    A PyTorch model wrapper for bird species classification.

    This class wraps a backbone from the `timm` library, replacing its default
    pooling and classification head with a custom Global Average Pooling (GAP)
    and Linear layer suitable for multi-label classification.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Args:
            model_name (str): The name of the timm model architecture (e.g., 'resnet34').
            num_classes (int): The number of output classes.
            pretrained (bool): Whether to load ImageNet pre-trained weights.
        """
        super(BirdModel, self).__init__()

        # Initialize the backbone.
        # num_classes=0 removes the original FC layer.
        # global_pool='' removes the original pooling, returning feature maps (B, C, H, W).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="", in_chans=3
        )

        # Determine the number of input channels for the head.
        if hasattr(self.backbone, "num_features"):
            in_features = self.backbone.num_features
        else:
            # Fallback: run a dummy forward pass to infer shape
            with torch.no_grad():
                dummy = torch.zeros(1, 3, 224, 224)
                features = self.backbone(dummy)
                in_features = features.shape[1]

        # Custom Head
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        x = self.backbone(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def get_model(
    model_name=Config.STUDENT_ARCH, num_classes=Config.NUM_CLASSES, pretrained=True
):
    """
    Factory function to create a BirdModel instance.

    Args:
        model_name (str): Name of the architecture (default from Config).
        num_classes (int): Number of classes (default from Config).
        pretrained (bool): Whether to use pretrained weights (default: True).

    Returns:
        BirdModel: The initialized model.
    """
    model = BirdModel(
        model_name=model_name, num_classes=num_classes, pretrained=pretrained
    )
    return model

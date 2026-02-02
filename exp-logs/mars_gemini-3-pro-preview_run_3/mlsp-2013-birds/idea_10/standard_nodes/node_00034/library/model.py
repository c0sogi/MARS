import torch
import torch.nn as nn
import timm


class BirdClassifier(nn.Module):
    """
    Bird Species Classifier using a pretrained backbone from the timm library.

    This class wraps standard architectures (like ResNet and EfficientNet) to adapt them
    for the specific requirements of the bird classification task:
    1. Accepts 3-channel RGB inputs (spectrograms are replicated across channels).
    2. Outputs logits for 19 specific bird species.
    3. Uses ImageNet pretrained weights for transfer learning.
    """

    def __init__(self, model_name, num_classes, pretrained=True):
        """
        Initialize the classifier.

        Args:
            model_name (str): Name of the architecture (e.g., 'resnet18', 'efficientnet_b0').
            num_classes (int): Number of output classes (species).
            pretrained (bool): If True, initializes with ImageNet weights.
        """
        super(BirdClassifier, self).__init__()

        # Instantiate the model using timm
        # in_chans=3: Explicitly expects 3 input channels (RGB).
        # num_classes: Replaces the original ImageNet head with a new linear layer for our classes.
        # global_pool='avg': Ensures global average pooling is applied before the classifier.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=3,
            global_pool="avg",
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch, num_classes).
        """
        return self.backbone(x)


def get_model(cfg, model_name):
    """
    Factory function to create a model instance based on configuration and model name.

    Args:
        cfg: Configuration object containing 'num_classes' and 'pretrained' settings.
        model_name (str): The specific architecture to instantiate (e.g., 'resnet18').

    Returns:
        nn.Module: An instance of BirdClassifier.
    """
    model = BirdClassifier(
        model_name=model_name, num_classes=cfg.num_classes, pretrained=cfg.pretrained
    )

    return model

import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class BirdResNet(nn.Module):
    """
    A ResNet-based architecture modified for multi-label bird species classification.
    """

    def __init__(
        self,
        model_name="resnet34",
        pretrained=True,
        num_classes=19,
        dropout_rate=0.0,
    ):
        """
        Args:
            model_name (str): Name of the ResNet backbone (currently supports 'resnet34').
            pretrained (bool): Whether to load ImageNet pre-trained weights.
            num_classes (int): Number of target classes (species).
            dropout_rate (float): Probability of dropout before the final layer.
        """
        super(BirdResNet, self).__init__()

        # Determine weights parameter for torchvision
        weights = None
        if model_name == "resnet34":
            if pretrained:
                weights = models.ResNet34_Weights.DEFAULT
            self.backbone = models.resnet34(weights=weights)
        else:
            # Fallback or extension point for other models
            raise ValueError(f"Model '{model_name}' is not currently supported.")

        # Replace the final fully connected layer
        # ResNet's final layer is named 'fc' and takes 'in_features'
        in_features = self.backbone.fc.in_features

        if dropout_rate > 0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
            )
        else:
            self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes).
        """
        return self.backbone(x)


def get_model(
    model_name=Config.MODEL_NAME,
    pretrained=Config.PRETRAINED,
    num_classes=Config.NUM_CLASSES,
    dropout_rate=Config.DROPOUT_RATE,
    freeze_backbone=False,
):
    """
    Factory function to initialize the model.

    Args:
        model_name (str): Name of the model architecture.
        pretrained (bool): Whether to use pre-trained weights.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout rate.
        freeze_backbone (bool): If True, freezes the feature extractor layers.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    model = BirdResNet(
        model_name=model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )

    if freeze_backbone:
        # Freeze all parameters in the backbone
        for name, param in model.backbone.named_parameters():
            # We only want to train the new head (fc layer)
            if "fc" not in name:
                param.requires_grad = False

    return model

import torch
import torch.nn as nn
import timm
from library.config import Config


class SEResNet34(nn.Module):
    """
    SE-ResNet34 architecture for multi-label bird species classification.

    This model utilizes a Squeeze-and-Excitation ResNet-34 backbone initialized
    with ImageNet weights. The classification head is a simple Linear layer
    projecting the global average pooled features to the 19 species classes.

    Attributes:
        backbone (nn.Module): The pretrained feature extractor.
        fc (nn.Linear): The classification head.
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        """
        Initialize the SEResNet34 model.

        Args:
            pretrained (bool): Whether to load pretrained ImageNet weights.
                               Defaults to Config.PRETRAINED.
        """
        super(SEResNet34, self).__init__()

        # Load the SE-ResNet34 backbone using timm
        # num_classes=0 removes the default classification head and returns pooled features
        # global_pool='avg' ensures the output is a flattened feature vector (B, num_features)
        self.backbone = timm.create_model(
            Config.BACKBONE_NAME,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        # Retrieve the number of input features for the linear layer
        # For SE-ResNet34, this is typically 512
        in_features = self.backbone.num_features

        # Define the simple linear classification head
        # Projects features to the number of bird species (19)
        self.fc = nn.Linear(in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.
                              Shape: (Batch, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH)

        Returns:
            torch.Tensor: Raw logits for each class.
                          Shape: (Batch, Config.NUM_CLASSES)
        """
        # Extract features using the backbone
        # Output shape: (Batch, in_features)
        features = self.backbone(x)

        # Project features to class logits
        # Output shape: (Batch, Config.NUM_CLASSES)
        logits = self.fc(features)

        return logits

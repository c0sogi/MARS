import torch
import torch.nn as nn
import timm
from library.config import Config


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier using an EfficientNet backbone.

    Attributes:
        backbone (nn.Module): The pre-trained feature extractor (EfficientNet-B0).
        fc (nn.Linear): The classification head.
    """

    def __init__(
        self,
        model_name="efficientnet_b0",
        pretrained=True,
        num_classes=Config.NUM_CLASSES,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): The name of the timm model to load.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
            num_classes (int): The number of output classes.
        """
        super(CassavaClassifier, self).__init__()

        # Load the backbone model using timm
        # num_classes=0 removes the original fully connected head.
        # global_pool='avg' ensures the output is a pooled feature vector (Global Average Pooling).
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Retrieve the number of input features for the classification head
        in_features = self.backbone.num_features

        # Define the new classification head (Single Fully Connected Layer)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images.

        Returns:
            torch.Tensor: Raw logits for each class.
        """
        # Extract features using the backbone (includes Global Average Pooling)
        features = self.backbone(x)

        # Pass features through the classification head
        logits = self.fc(features)

        return logits

    def freeze_backbone(self):
        """
        Freezes the parameters of the backbone network.
        Useful for the initial training epoch to stabilize the new head.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """
        Unfreezes the parameters of the backbone network.
        Useful for fine-tuning the entire model.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

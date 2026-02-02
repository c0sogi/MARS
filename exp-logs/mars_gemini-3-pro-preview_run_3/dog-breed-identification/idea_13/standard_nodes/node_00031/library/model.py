import torch
import torch.nn as nn
import timm
from library.config import Config


class DogClassifier(nn.Module):
    """
    Dog Breed Classifier using a ConvNeXt-Small backbone initialized with
    ImageNet-21k weights (fine-tuned on 1k).

    Attributes:
        backbone (nn.Module): The feature extractor.
        head (nn.Module): The classification head (Linear layer).
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=Config.pretrained,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the model architecture in timm.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(DogClassifier, self).__init__()

        # Load backbone with pooled features (num_classes=0 removes the default head)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0
        )

        # Get the number of input features for the head
        # ConvNeXt models typically have a 'num_features' attribute
        in_features = self.backbone.num_features

        # Define the custom classification head
        # As per strategy: Simple linear layer initialized randomly
        self.head = nn.Linear(in_features, num_classes)

        # Explicit initialization for the head (optional, but good practice)
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        features = self.backbone(x)
        logits = self.head(features)
        return logits

    def freeze_backbone(self):
        """
        Freezes the parameters of the backbone for the warm-up phase.
        """
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Ensure the head remains trainable
        for param in self.head.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes the parameters of the backbone for the fine-tuning phase.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_optimizer_params(self, learning_rate):
        """
        Helper to get parameters for the optimizer, potentially with different
        learning rates for backbone and head if needed.

        For this strategy, we use a uniform LR, but this structure allows flexibility.
        """
        return [
            {"params": self.backbone.parameters(), "lr": learning_rate},
            {"params": self.head.parameters(), "lr": learning_rate},
        ]

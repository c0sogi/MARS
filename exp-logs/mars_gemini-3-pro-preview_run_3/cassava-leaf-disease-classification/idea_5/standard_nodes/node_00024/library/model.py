import torch
import torch.nn as nn
import timm
from library.config import CFG


class CassavaClassifier(nn.Module):
    """
    Cassava Leaf Disease Classifier using a CoAtNet-2 backbone.
    """

    def __init__(
        self,
        model_name=CFG.model_name,
        pretrained=True,
        num_classes=CFG.num_classes,
        img_size=CFG.img_size,
    ):
        """
        Initializes the model architecture.

        Args:
            model_name (str): Name of the model backbone in timm.
            pretrained (bool): Whether to use pretrained ImageNet weights.
            num_classes (int): Number of target classes.
            img_size (int): Input image size (height/width).
        """
        super(CassavaClassifier, self).__init__()

        # Initialize the backbone using timm
        # We explicitly pass img_size to handle potential resolution differences
        # between pre-training (224) and fine-tuning (384), ensuring correct
        # interpolation of position embeddings if necessary.
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
            img_size=img_size,
        )

    def forward(self, x):
        """
        Forward pass of the neural network.

        Args:
            x (torch.Tensor): Input batch of images [B, C, H, W].

        Returns:
            torch.Tensor: Raw logits [B, num_classes].
        """
        return self.model(x)

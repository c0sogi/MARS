import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import seed_everything

# Ensure reproducibility by setting fixed seeds
seed_everything(Config.seed)


class DogBreedModel(nn.Module):
    """
    Dog Breed Prediction Model.

    This model uses a ConvNeXt-Small backbone pre-trained on ImageNet-21k
    (fine-tuned on ImageNet-1k) and replaces the classifier head with a
    linear layer for the 120 dog breed classes.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the model architecture.

        Args:
            pretrained (bool): Whether to load pre-trained weights for the backbone.
                               Defaults to True.
        """
        super(DogBreedModel, self).__init__()

        # Load the backbone using timm
        # Config.model_name is 'convnext_small.in22k_ft_in1k'
        # num_classes=0 removes the original head
        # global_pool='avg' ensures we get a flattened feature vector (B, num_features)
        self.backbone = timm.create_model(
            Config.model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        # Get the number of output features from the backbone
        in_features = self.backbone.num_features

        # Create the new classification head
        # Maps features to the 120 breed classes
        self.head = nn.Linear(in_features, Config.num_classes)

        # Initialize the head weights randomly
        # Using Xavier Uniform initialization for the weights and zeros for bias
        nn.init.xavier_uniform_(self.head.weight)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, Channels, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, Num_Classes).
        """
        # Pass input through the backbone to get features
        features = self.backbone(x)

        # Pass features through the classification head to get logits
        logits = self.head(features)

        return logits

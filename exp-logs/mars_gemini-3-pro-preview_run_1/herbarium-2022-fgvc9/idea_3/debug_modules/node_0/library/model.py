import torch
import torch.nn as nn
import timm
from library.config import Config


class HierarchicalConvNeXt(nn.Module):
    """
    Hierarchical ConvNeXt-Base model for Plant Classification.

    This model uses a pre-trained ConvNeXt backbone to extract features
    and employs three parallel classification heads to predict:
    1. Species (Fine-grained)
    2. Genus (Coarse-grained)
    3. Family (High-level taxonomy)
    """

    def __init__(self, model_name=None, pretrained=True):
        """
        Initialize the model.

        Args:
            model_name (str, optional): Name of the timm model to use.
                                        Defaults to Config.MODEL_NAME.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
        """
        super(HierarchicalConvNeXt, self).__init__()

        target_model = model_name if model_name else Config.MODEL_NAME

        # Create the backbone
        # num_classes=0 removes the default classifier and returns the pooled feature vector
        self.backbone = timm.create_model(
            target_model, pretrained=pretrained, num_classes=0
        )

        # Get the number of input features for the classification heads
        # ConvNeXt-Base usually has 1024 features
        n_features = self.backbone.num_features

        # Define hierarchical classification heads
        self.head_species = nn.Linear(n_features, Config.NUM_CLASSES)
        self.head_genus = nn.Linear(n_features, Config.NUM_GENERA)
        self.head_family = nn.Linear(n_features, Config.NUM_FAMILIES)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W).

        Returns:
            dict: A dictionary containing logits for each taxonomic level:
                  {
                      'species': Tensor (B, NUM_CLASSES),
                      'genus': Tensor (B, NUM_GENERA),
                      'family': Tensor (B, NUM_FAMILIES)
                  }
        """
        # Extract features from the backbone
        # Shape: (Batch_Size, n_features)
        features = self.backbone(x)

        # Pass features through parallel heads
        logits_species = self.head_species(features)
        logits_genus = self.head_genus(features)
        logits_family = self.head_family(features)

        return {
            "species": logits_species,
            "genus": logits_genus,
            "family": logits_family,
        }

import torch
import torch.nn as nn
import timm
from library.custom_layers import GeM, MultiSampleDropout


class PetModel(nn.Module):
    """
    A wrapper class that constructs the model architecture based on the provided configuration.
    It supports replacing the global pooling layer with Generalized Mean Pooling (GeM)
    or the classification head with Multi-Sample Dropout (MSD).
    """

    def __init__(
        self,
        model_name,
        num_classes=1,
        pretrained=True,
        use_gem=False,
        use_msd=False,
    ):
        """
        Args:
            model_name (str): The name of the backbone in timm.
            num_classes (int): Number of output classes (1 for binary classification).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            use_gem (bool): If True, replaces Global Average Pooling with GeM.
            use_msd (bool): If True, replaces the Linear head with Multi-Sample Dropout.
        """
        super(PetModel, self).__init__()
        self.use_gem = use_gem
        self.use_msd = use_msd
        self.model_name = model_name

        # ---------------------------------------------------------------------
        # Case 1: GeM Pooling (Target: ResNet, ConvNeXt)
        # ---------------------------------------------------------------------
        if use_gem:
            # Load backbone without pooling and head to get spatial features (B, C, H, W)
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
            )

            # Initialize GeM Pooling and Flatten layer
            self.gem = GeM()
            self.flatten = nn.Flatten()

            # Dynamically determine input features for the new head
            in_features = self.backbone.num_features
            self.head = nn.Linear(in_features, num_classes)

        # ---------------------------------------------------------------------
        # Case 2: Multi-Sample Dropout (Target: MaxViT)
        # ---------------------------------------------------------------------
        elif use_msd:
            # Load backbone with default pooling but no classification head
            # Returns pooled features (B, C)
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # Removes the linear head
            )

            in_features = self.backbone.num_features
            self.head = MultiSampleDropout(in_features, num_classes)

        # ---------------------------------------------------------------------
        # Case 3: Standard Architecture
        # ---------------------------------------------------------------------
        else:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
            )
            self.head = None

    def forward(self, x):
        """
        Forward pass of the model.
        """
        features = self.backbone(x)

        if self.use_gem:
            # features: (B, C, H, W)
            x = self.gem(features)  # (B, C, 1, 1)
            x = self.flatten(x)  # (B, C)
            return self.head(x)  # (B, num_classes)

        elif self.use_msd:
            # features: (B, C) - already pooled by backbone
            return self.head(features)  # (B, num_classes)

        else:
            # features: (B, num_classes) - backbone includes head
            return features


def build_model(
    model_name, num_classes=1, pretrained=True, use_gem=False, use_msd=False
):
    """
    Factory function to instantiate the PetModel.

    Args:
        model_name (str): Name of the timm model.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to use pretrained weights.
        use_gem (bool): Enable GeM pooling.
        use_msd (bool): Enable Multi-Sample Dropout.

    Returns:
        nn.Module: The configured PyTorch model.
    """
    return PetModel(
        model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        use_gem=use_gem,
        use_msd=use_msd,
    )

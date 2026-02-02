import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import log_message


class DogClassifier(nn.Module):
    """
    Dog Breed Classifier using ConvNeXt-Base backbone.
    Implements custom head with Dropout and utilities for discriminative learning rates.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
        dropout_rate=0.5,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pre-trained ImageNet weights.
            dropout_rate (float): Probability for the Dropout layer in the head.
        """
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes

        # Load the backbone model
        # We initialize with num_classes to ensure standard structure, then modify the head.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        # Modify the classification head
        # In timm's ConvNeXt implementation, the head block is 'head' and the linear layer is 'head.fc'
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            in_features = self.backbone.head.fc.in_features
            self.backbone.head.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
            )
        # Fallback for other common architectures (ResNet, EfficientNet) if model_name changes
        elif hasattr(self.backbone, "fc"):
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
            )
        elif hasattr(self.backbone, "classifier"):
            # Handle EfficientNet/MobileNet style
            if isinstance(self.backbone.classifier, nn.Linear):
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(p=dropout_rate), nn.Linear(in_features, num_classes)
                )
            else:
                # If classifier is already a Sequential, we assume the last layer is Linear
                # This is a generic fallback; specific handling is preferred
                pass
        else:
            log_message(
                f"Warning: Could not identify standard head for {model_name}. Model structure might be unexpected."
            )

    def forward(self, x):
        """
        Forward pass of the network.
        """
        return self.backbone(x)

    def _get_head_parameters(self):
        """
        Internal helper to retrieve the parameters of the modified classification head.
        """
        if hasattr(self.backbone, "head") and hasattr(self.backbone.head, "fc"):
            return self.backbone.head.fc.parameters()
        elif hasattr(self.backbone, "fc"):
            return self.backbone.fc.parameters()
        elif hasattr(self.backbone, "classifier"):
            return self.backbone.classifier.parameters()
        return []

    def freeze_backbone(self, freeze=True):
        """
        Freezes or unfreezes the backbone parameters.
        The classification head is always kept unfrozen.

        Args:
            freeze (bool): If True, sets requires_grad=False for backbone parameters.
        """
        # Identify head parameters by ID to exclude them from freezing
        head_params = list(self._get_head_parameters())
        head_ids = set(id(p) for p in head_params)

        for param in self.backbone.parameters():
            if id(param) not in head_ids:
                param.requires_grad = not freeze

        status = "Frozen" if freeze else "Unfrozen"
        # log_message(f"Backbone parameters {status}.")

    def get_optimizer_params(self, lr_backbone, lr_head):
        """
        Returns parameter groups for the optimizer to support discriminative learning rates.

        Args:
            lr_backbone (float): Learning rate for the pre-trained backbone.
            lr_head (float): Learning rate for the new classification head.

        Returns:
            list: A list of dictionaries defining parameter groups.
        """
        head_params = list(self._get_head_parameters())
        head_ids = set(id(p) for p in head_params)

        backbone_params = []
        for param in self.backbone.parameters():
            if param.requires_grad and id(param) not in head_ids:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]

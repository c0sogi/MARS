import torch
import torch.nn as nn
import timm
from library.config import Config


class BreedClassifier(nn.Module):
    """
    Dog Breed Classifier using ConvNeXt-Small backbone.

    Implements helper methods to support the Two-Phase Transfer Learning strategy:
    1. freeze_backbone(): For high-LR linear probing (head alignment).
    2. unfreeze_backbone(): For low-LR full fine-tuning.
    """

    def __init__(
        self, model_name: str = None, num_classes: int = None, pretrained: bool = True
    ):
        """
        Args:
            model_name (str): Name of the timm model to load. Defaults to Config.MODEL_NAME.
            num_classes (int): Number of output classes. Defaults to Config.NUM_CLASSES.
            pretrained (bool): Whether to load pre-trained weights. Defaults to True.
        """
        super().__init__()

        self.model_name = model_name if model_name else Config.MODEL_NAME
        self.num_classes = num_classes if num_classes else Config.NUM_CLASSES

        # Create the model using timm
        # When num_classes differs from the pretrained weights (e.g. 120 vs 1000),
        # timm automatically resets the classifier head to a random initialization
        # with the correct output dimension.
        self.backbone = timm.create_model(
            self.model_name, pretrained=pretrained, num_classes=self.num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        """
        return self.backbone(x)

    def freeze_backbone(self):
        """
        Freezes all parameters in the backbone, leaving only the classification head trainable.
        This is used for the 'Warmup' phase of training to align the random head
        with the pre-trained features.
        """
        # First, freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Then, unfreeze the classifier head
        # timm models expose the classifier via get_classifier()
        classifier = self.backbone.get_classifier()
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
        else:
            # Fallback if get_classifier is not available (unlikely for ConvNeXt)
            print("Warning: Could not identify classifier head to unfreeze.")

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters in the model.
        This is used for the 'Fine-tuning' phase of training.
        """
        for param in self.backbone.parameters():
            param.requires_grad = True

    def get_optimizer_params(self, lr_backbone: float, lr_head: float):
        """
        Helper to get parameter groups with different learning rates.
        Useful if one wants to use discriminative learning rates (Layer-Wise Decay),
        though the current strategy uses a global LR for phase 2.

        Args:
            lr_backbone: Learning rate for feature extractor layers.
            lr_head: Learning rate for the classifier head.
        """
        head_params = list(self.backbone.get_classifier().parameters())
        head_ids = list(map(id, head_params))

        backbone_params = [
            p for p in self.backbone.parameters() if id(p) not in head_ids
        ]

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": head_params, "lr": lr_head},
        ]

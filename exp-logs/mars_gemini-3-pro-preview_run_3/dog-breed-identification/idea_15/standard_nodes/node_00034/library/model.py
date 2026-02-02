import torch
import torch.nn as nn
import timm
from library.config import Config


class DogModel(nn.Module):
    """
    Dog Breed Classification Model based on ConvNeXt-Small.

    Attributes:
        model (nn.Module): The backbone model (ConvNeXt) with a custom classification head.
    """

    def __init__(self, pretrained=True):
        """
        Initializes the model.

        Args:
            pretrained (bool): Whether to load ImageNet-21k pre-trained weights.
                               Defaults to True.
        """
        super(DogModel, self).__init__()

        # Load ConvNeXt-Small pre-trained on ImageNet-21k
        # num_classes=Config.NUM_CLASSES ensures the head is replaced and randomly initialized
        # for our specific 120 classes.
        self.model = timm.create_model(
            Config.MODEL_NAME, pretrained=pretrained, num_classes=Config.NUM_CLASSES
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input batch of images (B, C, H, W).

        Returns:
            torch.Tensor: Logits (B, NUM_CLASSES).
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes the backbone parameters for Phase 1 (Head Warmup).
        Only the classification head parameters remain trainable.
        """
        # 1. Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # 2. Unfreeze the classification head
        # In timm's ConvNeXt implementation, the classifier is named 'head'
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "fc"):
            # Fallback for other architectures if config changes, though ConvNeXt uses 'head'
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            # Generic fallback: unfreeze parameters named 'head' or 'fc'
            for name, param in self.model.named_parameters():
                if "head" in name or "fc" in name:
                    param.requires_grad = True

        print("Backbone frozen. Only classification head is trainable.")

    def unfreeze_all(self):
        """
        Unfreezes all parameters for Phase 2 (Fine-Tuning).
        """
        for param in self.model.parameters():
            param.requires_grad = True

        print("All model parameters unfrozen for fine-tuning.")

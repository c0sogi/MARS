import torch
import torch.nn as nn
import timm


class HerbariumEfficientNet(nn.Module):
    """
    EfficientNet-B0 based model for Herbarium 2021 classification.

    Attributes:
        model (nn.Module): The EfficientNet-B0 backbone with a custom classifier head.
    """

    def __init__(self, num_classes):
        """
        Initializes the model.

        Args:
            num_classes (int): The number of output classes (species).
        """
        super(HerbariumEfficientNet, self).__init__()

        # Load EfficientNet-B0 pre-trained on ImageNet
        # Setting num_classes replaces the default 1000-class head with a new Linear layer
        self.model = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input images tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits of shape (B, num_classes).
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes the backbone parameters, leaving only the classifier head trainable.
        Used for the classifier re-balancing stage (Stage 2).
        """
        # Freeze all parameters in the model
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        # In timm's EfficientNet implementation, the head is named 'classifier'
        if hasattr(self.model, "classifier"):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            # Fallback check for other potential naming conventions (e.g. 'fc' in ResNet)
            # though efficientnet_b0 in timm strictly uses 'classifier'
            for name, param in self.model.named_parameters():
                if "classifier" in name or "fc" in name:
                    param.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreezes all parameters in the model.
        Used for the representation learning stage (Stage 1) or fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True

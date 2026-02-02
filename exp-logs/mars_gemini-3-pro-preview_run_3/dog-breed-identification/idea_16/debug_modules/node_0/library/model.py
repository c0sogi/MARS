import torch
import torch.nn as nn
import timm
from library.config import Config


class DogClassifier(nn.Module):
    """
    DogClassifier wraps a timm backbone (ConvNeXt-Small) for dog breed classification.
    It supports specific freezing/unfreezing logic for the two-phase transfer learning strategy.
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=True,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(DogClassifier, self).__init__()

        # Create the model using timm
        # num_classes ensures the head is replaced with a Linear layer of the correct size
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the network.
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes all parameters in the network except for the classification head.
        Used during the 'Head Warmup' phase (Phase 1).
        """
        # In timm models, the classifier is typically accessed via get_classifier()
        # or named 'head' (ConvNeXt) or 'fc' (ResNet).
        # We can identify head parameters by checking if they belong to the classifier module.

        classifier = self.model.get_classifier()

        # Iterate through all parameters
        for name, param in self.model.named_parameters():
            # Check if this parameter is part of the classifier
            is_classifier_param = False
            for cls_param in classifier.parameters():
                if param is cls_param:
                    is_classifier_param = True
                    break

            if is_classifier_param:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreezes all parameters in the network.
        Used during the 'Full Fine-tuning' phase (Phase 2).
        """
        for param in self.model.parameters():
            param.requires_grad = True

    def get_optimizer_params(self, lr_backbone, lr_head):
        """
        Returns parameter groups for the optimizer, allowing different learning rates
        for the backbone and the head if needed.

        Args:
            lr_backbone (float): Learning rate for backbone layers.
            lr_head (float): Learning rate for the head.

        Returns:
            list: List of dicts defining parameter groups.
        """
        classifier = self.model.get_classifier()
        classifier_params = list(classifier.parameters())
        classifier_ids = list(map(id, classifier_params))

        backbone_params = [
            p for p in self.model.parameters() if id(p) not in classifier_ids
        ]

        return [
            {"params": backbone_params, "lr": lr_backbone},
            {"params": classifier_params, "lr": lr_head},
        ]

import torch
import torch.nn as nn
import timm
from library.config import Config


class DogBreedModel(nn.Module):
    """
    Dog Breed Classification Model wrapper around a timm backbone.

    Architecture: ConvNeXt-Small (pre-trained on ImageNet-21k, fine-tuned on 1k).
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of target classes (breeds).
            pretrained (bool): Whether to load pre-trained weights.
        """
        super(DogBreedModel, self).__init__()

        # Create the model using timm
        # This handles the download of weights and replacement of the classification head
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the model.
        """
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes the backbone parameters for the warm-up phase.
        Only the classification head remains trainable.
        """
        # First, freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Retrieve the classifier head using timm's standard API
        classifier = self.model.get_classifier()

        # Unfreeze the classifier parameters
        if classifier is not None:
            for param in classifier.parameters():
                param.requires_grad = True
        else:
            # Fallback: manually search for common head names if get_classifier fails
            # (ConvNeXt uses 'head', others use 'fc' or 'classifier')
            for name, param in self.model.named_parameters():
                if any(x in name for x in ["head", "fc", "classifier"]):
                    param.requires_grad = True

    def unfreeze_backbone(self):
        """
        Unfreezes all parameters for the fine-tuning phase.
        """
        for param in self.model.parameters():
            param.requires_grad = True


def get_model(device=Config.DEVICE, pretrained=True):
    """
    Factory function to create the model and move it to the configured device.

    Args:
        device (str): The device to move the model to (e.g., 'cuda', 'cpu').
        pretrained (bool): Whether to initialize with pre-trained weights.

    Returns:
        DogBreedModel: The initialized model instance.
    """
    model = DogBreedModel(pretrained=pretrained)
    model.to(device)
    return model

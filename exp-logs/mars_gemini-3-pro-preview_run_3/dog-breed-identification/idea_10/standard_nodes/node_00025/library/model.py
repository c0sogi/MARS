import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

# Initialize logger
logger = get_logger("model")


class DogModel(nn.Module):
    """
    Wrapper class for the Dog Breed Classification model based on ConvNeXt.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Initializes the model.

        Args:
            model_name (str): Name of the architecture in timm.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
        """
        super(DogModel, self).__init__()

        logger.info(
            f"Initializing model: {model_name} | Pretrained: {pretrained} | Classes: {num_classes}"
        )

        # Create the model using timm
        # num_classes ensures the head is replaced/initialized correctly for our task
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        """
        Forward pass of the network.
        """
        return self.backbone(x)

    def set_backbone_trainable(self, trainable: bool):
        """
        Freezes or unfreezes the backbone parameters for transfer learning.

        Args:
            trainable (bool): If True, all parameters are trainable (Fine-tuning).
                              If False, only the head is trainable (Warm-up).
        """
        if trainable:
            logger.info("Unfreezing backbone. All parameters are now trainable.")
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            logger.info("Freezing backbone. Only the classification head is trainable.")
            # First, freeze everything
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Then, unfreeze the head
            # ConvNeXt usually uses 'head' as the final layer name
            # We iterate through named parameters to be robust
            head_found = False
            for name, param in self.backbone.named_parameters():
                # Common names for classification heads in timm models
                if "head" in name or "fc" in name or "classifier" in name:
                    param.requires_grad = True
                    head_found = True

            if not head_found:
                logger.warning(
                    "Could not identify classification head by name. "
                    "Please check model architecture. Backbone might be fully frozen."
                )

    def get_param_count(self):
        """
        Returns the number of trainable and total parameters.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return trainable_params, total_params


def get_model(device=Config.DEVICE, pretrained=True):
    """
    Factory function to create the model and move it to the configured device.

    Args:
        device (torch.device): The device to move the model to.
        pretrained (bool): Whether to use pretrained weights.

    Returns:
        DogModel: The instantiated model.
    """
    model = DogModel(pretrained=pretrained)
    model.to(device)

    trainable, total = model.get_param_count()
    logger.info(
        f"Model created. Trainable params: {trainable:,} / Total params: {total:,}"
    )

    return model

import torch
import torch.nn as nn
import timm
import copy
from library.config import Config


class DogModel(nn.Module):
    """
    Dog Breed Classification Model based on ConvNeXt architecture.
    Wraps timm.create_model to handle backbone loading and head replacement.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=True,
    ):
        """
        Args:
            model_name (str): Name of the model architecture in timm.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load ImageNet-22k pretrained weights.
        """
        super(DogModel, self).__init__()

        # Create the model using timm
        # Setting num_classes automatically replaces the head with a Linear layer of the correct size.
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)

    def freeze_backbone(self):
        """
        Freezes all parameters except the classifier head.
        Used for the first phase of transfer learning (Head alignment).
        """
        # Freeze all parameters first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze the classifier head
        # In timm ConvNeXt implementation, the classifier is named 'head'
        if hasattr(self.model, "head"):
            for param in self.model.head.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "fc"):
            # Fallback for ResNet-like architectures
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif hasattr(self.model, "classifier"):
            # Fallback for EfficientNet/MobileNet-like architectures
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        else:
            # Generic fallback using timm's interface
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def unfreeze_all(self):
        """
        Unfreezes all parameters in the model.
        Used for the second phase of fine-tuning.
        """
        for param in self.model.parameters():
            param.requires_grad = True


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters for better generalization and stability.
    """

    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay factor for the moving average.
            device (torch.device): Device to store the shadow model on.
        """
        self.decay = decay
        # Create a deep copy of the model to serve as the shadow model
        self.module = copy.deepcopy(model)

        # Ensure the shadow model is in evaluation mode
        self.module.eval()

        # Determine device
        if device is None:
            # Try to infer device from the source model
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")

        self.device = device
        self.module.to(self.device)

        # Disable gradients for the shadow model to save memory
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the shadow model parameters using the current model parameters.
        Formula: shadow = decay * shadow + (1 - decay) * current
        """
        with torch.no_grad():
            msd = model.state_dict()
            shadow_sd = self.module.state_dict()

            for k, v in shadow_sd.items():
                if k in msd:
                    # Ensure the current parameter is on the correct device before computation
                    current_param = msd[k].to(self.device)
                    # In-place update
                    v.copy_(self.decay * v + (1.0 - self.decay) * current_param)

    def forward(self, x):
        """
        Forward pass using the shadow model.
        """
        return self.module(x)

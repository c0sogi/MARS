import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from copy import deepcopy
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.

    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp to avoid numerical instability with pow
        x = x.clamp(min=eps)
        # Average pooling on x^p
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class ArtworkModel(nn.Module):
    """
    Artwork Attribution Model based on ConvNeXt backbone.
    Replaces the default classifier with a GeM pooling layer and a custom Linear head.
    """

    def __init__(
        self,
        model_name=Config.model_name,
        num_classes=Config.num_classes,
        pretrained=Config.pretrained,
    ):
        super(ArtworkModel, self).__init__()

        # Initialize backbone
        self.backbone = timm.create_model(model_name, pretrained=pretrained)

        # Get the number of input features for the final layer
        self.in_features = self.backbone.num_features

        # Remove the original classification head and pooling
        # reset_classifier(0, '') ensures we get the spatial feature map (B, C, H, W)
        self.backbone.reset_classifier(0, "")

        # Pooling layer
        self.pooling = GeM()

        # Flatten layer
        self.flatten = nn.Flatten()

        # Classification head
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Extract features: (B, C, H, W)
        features = self.backbone.forward_features(x)

        # Apply GeM pooling: (B, C, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (B, C)
        flattened = self.flatten(pooled_features)

        # Classification: (B, num_classes)
        output = self.fc(flattened)

        return output


class ModelEMA:
    """
    Model Exponential Moving Average (EMA).
    Maintains a moving average of model parameters to improve robustness and generalization.
    """

    def __init__(self, model, decay=Config.ema_decay, device=None):
        """
        Args:
            model (nn.Module): The model to track.
            decay (float): The decay rate for the moving average.
            device (torch.device): Device to store the EMA model on.
        """
        self.module = deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device

        if self.device:
            self.module.to(device)

        # Freeze parameters of the EMA model
        for param in self.module.parameters():
            param.requires_grad = False

    def update(self, model):
        """
        Update the EMA model parameters based on the current model.

        Args:
            model (nn.Module): The current training model.
        """
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.module.state_dict()

            for k in msd.keys():
                model_v = msd[k].detach()
                ema_v = esd[k]

                if self.device:
                    model_v = model_v.to(self.device)

                # Update: ema_v = decay * ema_v + (1 - decay) * model_v
                ema_v.copy_(ema_v * self.decay + model_v * (1.0 - self.decay))

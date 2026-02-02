import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (B, C, H, W)
        # Clamp to avoid NaN in power operation
        x = x.clamp(min=self.eps)
        # Average pooling on x^p
        # The kernel size is the spatial size of the input (H, W)
        return F.avg_pool2d(x.pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"


class DRModel(nn.Module):
    """
    Diabetic Retinopathy Classification Model.
    Uses a ConvNeXt backbone with GeM pooling and a regression head.
    """

    def __init__(self, model_name=Config.BACKBONE, pretrained=True):
        super(DRModel, self).__init__()

        # Create backbone
        # num_classes=0 and global_pool="" ensures we get the spatial feature map (B, C, H, W)
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine the number of input features for the head
        self.in_features = self.backbone.num_features

        # Generalized Mean Pooling
        self.gem = GeM()

        # Regression Head
        # Maps features to a single continuous score
        self.head = nn.Linear(self.in_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
        Returns:
            torch.Tensor: Predicted scores of shape (B, 1)
        """
        # Extract features: (B, C, H, W)
        features = self.backbone(x)

        # Apply GeM pooling: (B, C, 1, 1)
        pooled = self.gem(features)

        # Flatten: (B, C)
        pooled = pooled.flatten(1)

        # Predict score: (B, 1)
        output = self.head(pooled)

        return output

    def set_grad_checkpointing(self, enable=True):
        """
        Enables or disables gradient checkpointing for the backbone.
        Useful for training with large image sizes to save memory.
        """
        if hasattr(self.backbone, "set_grad_checkpointing"):
            self.backbone.set_grad_checkpointing(enable)
        else:
            # Fallback for models that might not expose the method directly via timm wrapper
            # though most modern timm models do.
            print(
                f"Warning: Backbone {Config.BACKBONE} may not support explicit gradient checkpointing via set_grad_checkpointing."
            )

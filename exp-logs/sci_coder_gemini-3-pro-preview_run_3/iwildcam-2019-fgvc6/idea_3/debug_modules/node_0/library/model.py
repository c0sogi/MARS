import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of each channel in the feature map.
    Formula: f = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        """
        Args:
            p (float): Initial power parameter.
            eps (float): Epsilon for numerical stability.
        """
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).
        Returns:
            torch.Tensor: Pooled tensor of shape (B, C, 1, 1).
        """
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # Clamp for stability, raise to power p, avg pool, raise to power 1/p
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class AnimalSwinV2(nn.Module):
    """
    Swin Transformer V2 model for Animal Classification.
    Uses SwinV2 Tiny backbone with GeM pooling and a linear head.
    """

    def __init__(
        self,
        model_name=Config.MODEL_NAME,
        num_classes=Config.NUM_CLASSES,
        pretrained=Config.PRETRAINED,
        drop_path_rate=Config.DROP_PATH_RATE,
        use_gem=Config.USE_GEM_POOLING,
    ):
        """
        Args:
            model_name (str): Name of the timm model to load.
            num_classes (int): Number of output classes.
            pretrained (bool): Whether to load pretrained weights.
            drop_path_rate (float): Stochastic depth rate.
            use_gem (bool): Whether to use GeM pooling.
        """
        super(AnimalSwinV2, self).__init__()

        # Load backbone
        # global_pool='' ensures we get spatial features (B, H, W, C) for Swin
        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            drop_path_rate=drop_path_rate,
        )

        # Get feature dimension (e.g., 768 for Swin Tiny)
        self.num_features = self.backbone.num_features

        # Pooling Layer
        self.use_gem = use_gem
        if self.use_gem:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.head = nn.Linear(self.num_features, num_classes)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input images (B, C, H, W).
        Returns:
            torch.Tensor: Logits (B, Num_Classes).
        """
        # Backbone forward pass
        # Swin Transformer in timm returns features as (B, H, W, C)
        x = self.backbone(x)

        # Permute to (B, C, H, W) for pooling compatibility
        x = x.permute(0, 3, 1, 2)

        # Apply Pooling -> (B, C, 1, 1)
        x = self.pooling(x)

        # Flatten -> (B, C)
        x = x.view(x.size(0), -1)

        # Classification Head -> (B, Num_Classes)
        x = self.head(x)

        return x

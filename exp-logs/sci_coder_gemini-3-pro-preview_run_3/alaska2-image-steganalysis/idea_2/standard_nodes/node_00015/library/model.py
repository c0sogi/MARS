import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from library.config import Config
from library.utils import get_hpf_kernel


class HPFConv2d(nn.Module):
    """
    A custom Convolutional layer initialized with a fixed High-Pass Filter (KV Kernel).
    This layer acts as a preprocessing stem to extract noise residuals from the image.
    Cite solution_lesson_node_00001
    """

    def __init__(self, in_channels=1, out_channels=1):
        super(HPFConv2d, self).__init__()

        # Initialize Conv2d layer
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=5, stride=1, padding=2, bias=False
        )

        # Get fixed HPF kernel
        hpf_kernel = get_hpf_kernel()

        # Assign weights and freeze them
        self.conv.weight.data = hpf_kernel
        self.conv.weight.requires_grad = False

    def forward(self, x):
        return self.conv(x)


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM).
    Computes the p-th power average of the input, which is a generalization of
    Average Pooling (p=1) and Max Pooling (p -> infinity).
    The parameter p is trainable.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (N, C, H, W)
        # Clamp to avoid numerical instability with pow
        # Average pooling over spatial dimensions (H, W)
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


class BCEWithLogitsLossWithSmoothing(nn.Module):
    def __init__(self, label_smoothing=0.0):
        super(BCEWithLogitsLossWithSmoothing, self).__init__()
        self.label_smoothing = label_smoothing
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return self.bce(logits, targets)


class MonoResidualEfficientNet(nn.Module):
    """
    Steganalysis model combining a single HPF residual stem with an EfficientNet backbone.
    Uses Single-Channel Residual Adaptation (Cite solution_lesson_node_00009).
    """

    def __init__(self, backbone_name=Config.BACKBONE, pretrained=True):
        super(MonoResidualEfficientNet, self).__init__()

        # 1. HPF Stem
        # Extracts 1 channel of noise residuals
        self.hpf = HPFConv2d(
            in_channels=Config.SRM_IN_CHANNELS, out_channels=Config.SRM_OUT_CHANNELS
        )

        # 2. Backbone
        # We use timm to create the backbone.
        # in_chans=1 adapts the first conv layer (summing weights) to accept the HPF output.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=Config.SRM_OUT_CHANNELS,  # Should be 1
            num_classes=0,
            global_pool="",
        )

        # Determine the number of output features from the backbone
        with torch.no_grad():
            dummy_input = torch.randn(1, Config.SRM_OUT_CHANNELS, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # 3. Pooling Layer
        if Config.USE_GEM_POOLING:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # 4. Classification Head
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (N, 1, H, W).
        Returns:
            torch.Tensor: Logits (N, 1).
        """
        # Extract residuals
        x = self.hpf(x)  # (N, 1, H, W)

        # Extract features
        x = self.backbone(x)  # (N, C, H', W')

        # Pool features
        x = self.pooling(x)  # (N, C, 1, 1)

        # Flatten
        x = x.flatten(1)  # (N, C)

        # Classify
        x = self.fc(x)  # (N, 1)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Clamp to avoid NaN with log/pow operations
        x = x.clamp(min=eps)
        # Apply GeM formula: (AvgPool(x^p))^(1/p)
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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


class WhaleConvNeXt(nn.Module):
    """
    ConvNeXt-Tiny model adapted for 1-channel spectrogram input with GeM pooling.
    """

    def __init__(self):
        super(WhaleConvNeXt, self).__init__()

        # 1. Load Pretrained Backbone
        # We load with in_chans=3 first to ensure we get the standard pretrained weights,
        # then we will manually adapt the first layer.
        # num_classes=0 and global_pool='' removes the default head and pooling.
        self.backbone = timm.create_model(
            Config.MODEL_NAME,
            pretrained=Config.PRETRAINED,
            num_classes=0,
            global_pool="",
            in_chans=3,
        )

        # 2. Modify First Layer for 1-Channel Input
        # In ConvNeXt, the stem contains the first convolution.
        # Structure: backbone.stem[0] is the Conv2d layer.
        original_conv = self.backbone.stem[0]

        new_conv = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,  # 1
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=original_conv.bias is not None,
        )

        # Initialize new weights by averaging the original RGB weights
        with torch.no_grad():
            # original_conv.weight shape: (Out, 3, H, W)
            # new_conv.weight shape: (Out, 1, H, W)
            new_conv.weight[:] = torch.mean(original_conv.weight, dim=1, keepdim=True)
            if original_conv.bias is not None:
                new_conv.bias[:] = original_conv.bias

        # Replace the layer in the backbone
        self.backbone.stem[0] = new_conv

        # 3. Define Pooling
        if Config.USE_GEM_POOLING:
            self.pool = GeM()
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)

        # 4. Define Classifier Head
        # ConvNeXt-Tiny usually has 768 features
        self.num_features = self.backbone.num_features
        self.head = nn.Linear(self.num_features, Config.NUM_CLASSES)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input spectrograms (B, 1, H, W)
        Returns:
            x (Tensor): Logits (B, 1)
        """
        # Extract features from backbone
        # Output shape: (B, C, H_feat, W_feat)
        x = self.backbone.forward_features(x)

        # Apply Pooling
        # Output shape: (B, C, 1, 1)
        x = self.pool(x)

        # Flatten
        # Output shape: (B, C)
        x = x.flatten(1)

        # Classification
        # Output shape: (B, NUM_CLASSES)
        x = self.head(x)

        return x

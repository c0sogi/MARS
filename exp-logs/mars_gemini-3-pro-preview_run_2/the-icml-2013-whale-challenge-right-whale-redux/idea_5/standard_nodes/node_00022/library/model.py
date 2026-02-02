import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean: (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
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


class WhaleEfficientNet(nn.Module):
    """
    EfficientNet-B2 with GeM Pooling for Whale Call Detection.
    Accepts 3-channel input (Log-Mel, Delta, Delta-Delta).
    """

    def __init__(self, pretrained=Config.PRETRAINED):
        super(WhaleEfficientNet, self).__init__()

        # Load pretrained backbone
        if Config.IN_CHANNELS == 1 and pretrained:
            # Load with in_chans=3 to get original ImageNet weights
            self.backbone = timm.create_model(
                Config.MODEL_NAME,
                pretrained=pretrained,
                in_chans=3,
                num_classes=0,
                global_pool="",
            )
            # Adapt first layer (conv_stem for EfficientNet) by averaging weights
            # Cite solution_lesson_node_00001
            if hasattr(self.backbone, "conv_stem"):
                old_layer = self.backbone.conv_stem
                new_layer = nn.Conv2d(
                    1,
                    old_layer.out_channels,
                    kernel_size=old_layer.kernel_size,
                    stride=old_layer.stride,
                    padding=old_layer.padding,
                    bias=old_layer.bias is not None,
                )
                # Average the weights across the RGB channels
                new_layer.weight.data = old_layer.weight.data.mean(dim=1, keepdim=True)
                if old_layer.bias is not None:
                    new_layer.bias.data = old_layer.bias.data
                self.backbone.conv_stem = new_layer
        else:
            self.backbone = timm.create_model(
                Config.MODEL_NAME,
                pretrained=pretrained,
                in_chans=Config.IN_CHANNELS,
                num_classes=0,
                global_pool="",
            )

        # Determine the number of output features from the backbone
        self.num_features = self.backbone.num_features

        # Pooling layer
        if Config.USE_GEM:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Classification Head
        self.fc = nn.Linear(self.num_features, Config.NUM_CLASSES)

        # Initialize weights for the head
        self._init_weights(self.fc)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Extract features: (Batch, Channels, Height, Width)
        features = self.backbone.forward_features(x)

        # Apply Pooling: (Batch, Channels, 1, 1)
        pooled_features = self.pooling(features)

        # Flatten: (Batch, Channels)
        flattened_features = pooled_features.view(pooled_features.size(0), -1)

        # Classification
        logits = self.fc(flattened_features)

        return logits

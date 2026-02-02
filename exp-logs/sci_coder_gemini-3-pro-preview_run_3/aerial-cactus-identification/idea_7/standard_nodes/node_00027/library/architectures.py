import torch
import torch.nn as nn
import timm


class AdaptedTimmModel(nn.Module):
    """
    Wrapper for timm models to adapt them for 32x32 input images.
    It replaces the initial downsampling stem with a stride-1 convolution
    and removes early pooling layers to preserve spatial resolution.
    """

    def __init__(self, model_name, num_classes=1, pretrained=True):
        super(AdaptedTimmModel, self).__init__()
        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )

        # Adapt stem based on architecture type
        if "resnet" in model_name:
            # ResNet: conv1 (7x7 s2) -> bn1 -> act -> maxpool (3x3 s2)
            # We replace conv1 with 3x3 s1 and remove maxpool
            # This changes total downsampling from 4x to 1x at the stem

            # Check if conv1 exists (standard ResNet)
            if hasattr(self.model, "conv1"):
                in_c = self.model.conv1.in_channels
                out_c = self.model.conv1.out_channels
                self.model.conv1 = nn.Conv2d(
                    in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False
                )
                # Random initialization for the new layer (Cite Lesson 23)
                nn.init.kaiming_normal_(
                    self.model.conv1.weight, mode="fan_out", nonlinearity="relu"
                )

            # Remove maxpool
            if hasattr(self.model, "maxpool"):
                self.model.maxpool = nn.Identity()

        elif "densenet" in model_name:
            # DenseNet: features.conv0 (7x7 s2) -> norm0 -> relu0 -> pool0 (3x3 s2)
            # Replace conv0 with 3x3 s1, remove pool0
            if hasattr(self.model.features, "conv0"):
                in_c = self.model.features.conv0.in_channels
                out_c = self.model.features.conv0.out_channels
                self.model.features.conv0 = nn.Conv2d(
                    in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False
                )
                nn.init.kaiming_normal_(
                    self.model.features.conv0.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            if hasattr(self.model.features, "pool0"):
                self.model.features.pool0 = nn.Identity()

        elif "efficientnet" in model_name:
            # EfficientNet: conv_stem (3x3 s2)
            # Replace with 3x3 s1
            if hasattr(self.model, "conv_stem"):
                in_c = self.model.conv_stem.in_channels
                out_c = self.model.conv_stem.out_channels
                self.model.conv_stem = nn.Conv2d(
                    in_c, out_c, kernel_size=3, stride=1, padding=1, bias=False
                )
                nn.init.kaiming_normal_(
                    self.model.conv_stem.weight, mode="fan_out", nonlinearity="relu"
                )
                # EfficientNet typically does not have an explicit maxpool in the stem

    def forward(self, x):
        return self.model(x)


def get_model(model_name, num_classes=1, pretrained=True):
    """
    Factory function to instantiate adapted timm models.
    """
    return AdaptedTimmModel(model_name, num_classes=num_classes, pretrained=pretrained)

import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class IBN(nn.Module):
    """
    Instance-Batch Normalization Layer.
    Splits channels into two halves:
    - Half 1: Instance Normalization (Style Invariant)
    - Half 2: Batch Normalization (Content Preserving)
    """

    def __init__(self, planes):
        super(IBN, self).__init__()
        self.half = planes // 2
        # Affine=True allows IN to learn scale/shift, similar to BN
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        # Split input tensor along channel dimension
        split = torch.split(x, self.half, 1)

        # Apply normalizations
        out1 = self.IN(split[0].contiguous())
        out2 = self.BN(split[1].contiguous())

        # Concatenate back
        return torch.cat((out1, out2), 1)


def convert_resnet_to_ibn(model):
    """
    Modifies a standard ResNet50 to ResNet50-IBN-a.
    Replaces bn1 in Bottleneck blocks of layer1, layer2, and layer3 with IBN.
    Transfers pretrained weights from the original BN to the new IBN layer.
    """
    # IBN-a is applied to conv2_x, conv3_x, conv4_x (layers 1, 2, 3)
    target_layers = [model.layer1, model.layer2, model.layer3]

    for layer in target_layers:
        for block in layer:
            # In ResNet Bottleneck, bn1 follows the first 1x1 conv
            old_bn = block.bn1
            planes = old_bn.num_features

            # Create new IBN layer
            ibn = IBN(planes)

            # Transfer weights
            # The original BN has C channels. IBN splits into C/2 (IN) and C/2 (BN).
            half = planes // 2

            with torch.no_grad():
                # Copy first half to Instance Norm
                ibn.IN.weight.copy_(old_bn.weight[:half])
                ibn.IN.bias.copy_(old_bn.bias[:half])

                # Copy second half to Batch Norm
                ibn.BN.weight.copy_(old_bn.weight[half:])
                ibn.BN.bias.copy_(old_bn.bias[half:])
                ibn.BN.running_mean.copy_(old_bn.running_mean[half:])
                ibn.BN.running_var.copy_(old_bn.running_var[half:])

            # Replace the layer
            block.bn1 = ibn

    return model


class WhaleModel(nn.Module):
    def __init__(self, arch=None, num_classes=None, pretrained=True):
        """
        Args:
            arch (str): 'densenet121' or 'resnet50_ibn_a'.
            num_classes (int): Number of classes (unused for embeddings, but kept for API consistency).
            pretrained (bool): Whether to load ImageNet weights.
        """
        super(WhaleModel, self).__init__()
        self.arch = arch

        # ---------------------------------------------------------------------
        # Backbone Construction
        # ---------------------------------------------------------------------
        if arch == "densenet121":
            weights = "DEFAULT" if pretrained else None
            base = models.densenet121(weights=weights)
            self.features = base.features
            in_features = base.classifier.in_features  # 1024

        elif arch == "resnet50_ibn_a":
            weights = "DEFAULT" if pretrained else None
            base = models.resnet50(weights=weights)
            # Convert to IBN-Net
            base = convert_resnet_to_ibn(base)

            # ResNet structure is not a single sequential container like DenseNet.
            # We wrap the parts we need.
            self.base = base
            in_features = base.fc.in_features  # 2048

            # Remove the original FC layer to save memory/confusion
            del self.base.fc

        else:
            raise ValueError(f"Unknown architecture: {arch}")

        # ---------------------------------------------------------------------
        # Pooling & Neck
        # ---------------------------------------------------------------------
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Projection Head: Linear -> BN
        # No Dropout, No ReLU (optimized for ArcFace metric learning)
        self.neck = nn.Sequential(
            nn.Linear(in_features, Config.EMBEDDING_SIZE, bias=False),
            nn.BatchNorm1d(Config.EMBEDDING_SIZE),
        )

    def forward(self, x):
        # ---------------------------------------------------------------------
        # Feature Extraction
        # ---------------------------------------------------------------------
        if self.arch == "densenet121":
            x = self.features(x)
            # DenseNet features usually need a ReLU before pooling
            # (as the last block ends with BN)
            x = nn.ReLU(inplace=True)(x)
            x = self.pool(x)
            x = torch.flatten(x, 1)

        elif self.arch == "resnet50_ibn_a":
            # Manual forward pass for ResNet to skip the FC layer
            x = self.base.conv1(x)
            x = self.base.bn1(x)
            x = self.base.relu(x)
            x = self.base.maxpool(x)

            x = self.base.layer1(x)
            x = self.base.layer2(x)
            x = self.base.layer3(x)
            x = self.base.layer4(x)

            x = self.pool(x)
            x = torch.flatten(x, 1)

        # ---------------------------------------------------------------------
        # Projection
        # ---------------------------------------------------------------------
        embeddings = self.neck(x)

        return embeddings

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    ArcFace (Additive Angular Margin Loss) layer.
    """

    def __init__(self, in_features, out_features, s=64.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        # input: (batch, in_features)
        # cosine: (batch, out_features)
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            return cosine * self.s

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class HotelResNet(nn.Module):
    """
    A ResNet-18 based neural network for Hotel Identification.
    Uses ArcFace head for better discrimination of tail classes.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        super(HotelResNet, self).__init__()

        weights = "DEFAULT" if pretrained else None
        if Config.MODEL_NAME == "resnet50":
            self.backbone = models.resnet50(weights=weights)
        else:
            self.backbone = models.resnet18(weights=weights)

        # Replace fc with Identity to get features
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        # ArcFace Head
        self.arcface = ArcMarginProduct(in_features, num_classes)

    def forward(self, x, targets=None):
        """
        Forward pass.
        Args:
            x: Input images
            targets: Labels (optional, for training with ArcFace)
        """
        features = self.backbone(x)
        return self.arcface(features, targets)

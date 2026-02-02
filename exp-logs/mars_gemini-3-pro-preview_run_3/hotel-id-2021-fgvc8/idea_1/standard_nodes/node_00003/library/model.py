import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from library.config import Config


class ArcMarginProduct(nn.Module):
    """
    Implement of large margin cosine distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    Cite solution_lesson_node_00002: Use Angular Margin Loss for high-cardinality identification.
    """

    def __init__(
        self,
        in_features,
        out_features,
        s=30.0,
        m=0.50,
        easy_margin=False,
        device="cuda",
    ):
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
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        if label is None:
            # Inference mode: return scaled cosine similarities
            return cosine * self.s

        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # --------------------------- convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # -------------torch.where(out_i = {x_i if condition_i else y_i}) -------------
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output


class HotelResNet(nn.Module):
    """
    A ResNet-18 based neural network for Hotel Identification.
    Modified to use ArcFace head for better metric learning on high-cardinality data.
    """

    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED):
        """
        Initialize the HotelResNet model.
        """
        super(HotelResNet, self).__init__()

        # Select weights based on the pretrained flag
        weights = "DEFAULT" if pretrained else None

        # Load the ResNet-18 backbone
        self.backbone = models.resnet18(weights=weights)

        # Retrieve the number of input features for the fc layer (512 for ResNet18)
        in_features = self.backbone.fc.in_features

        # Replace the fc layer with Identity to get embeddings
        self.backbone.fc = nn.Identity()

        # Add ArcFace Head
        self.arc_head = ArcMarginProduct(in_features, num_classes)

    def forward(self, x, targets=None):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input batch of images.
            targets (torch.Tensor, optional): Ground truth labels for ArcFace margin.

        Returns:
            torch.Tensor: Logits (scaled cosine similarities).
        """
        # Get embeddings from backbone
        features = self.backbone(x)

        # Pass through ArcFace head
        return self.arc_head(features, targets)

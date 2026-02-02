import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from library.config import Config


class ArcMarginProduct(nn.Module):
    r"""
    Implement of large margin arc distance:
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        s: norm of input feature
        m: margin
        cos(theta + m)
    """

    def __init__(self, in_features, out_features, s=30.0, m=0.50, easy_margin=False):
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
        # L2 Normalize input features and weights
        # input: [batch_size, in_features]
        # weight: [out_features, in_features]
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))

        # If no label is provided (inference/validation), return scaled cosine similarities
        if label is None:
            return cosine * self.s

        # --------------------------- Training with Margin ---------------------------
        # Calculate sin(theta)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        # Calculate cos(theta + m) using angle sum identity
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # --------------------------- Convert label to one-hot ---------------------------
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        # --------------------------- Add margin to ground truth ---------------------------
        # For the true class: use phi (cos(theta+m))
        # For others: use cosine (cos(theta))
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # Scale the result
        output *= self.s

        return output


class ArcFaceResNet(nn.Module):
    def __init__(self):
        super(ArcFaceResNet, self).__init__()

        # Load ResNet18 backbone with default (ImageNet) weights
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Define embedding size
        self.embedding_size = Config.EMBEDDING_SIZE

        # Replace the original fully connected layer
        # ResNet18's fc input dimension is 512
        # We project this to the embedding size (e.g., 512)
        # Using a Linear + BatchNorm structure is standard for metric learning embeddings
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, self.embedding_size), nn.BatchNorm1d(self.embedding_size)
        )

        # ArcFace Head
        self.arcface = ArcMarginProduct(
            in_features=self.embedding_size,
            out_features=Config.NUM_CLASSES,
            s=Config.ARCFACE_SCALE,
            m=Config.ARCFACE_MARGIN,
        )

    def forward(self, images, labels=None):
        """
        Forward pass.

        Args:
            images (torch.Tensor): Input images.
            labels (torch.Tensor, optional): Ground truth labels.
                                           If provided, applies ArcFace margin (Training).
                                           If None, returns scaled cosine similarities (Inference).
        """
        # Extract features (embedding)
        # Shape: [batch_size, embedding_size]
        features = self.backbone(images)

        # Pass through ArcFace head
        # Shape: [batch_size, num_classes]
        logits = self.arcface(features, labels)

        return logits

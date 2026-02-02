import torch
import torch.nn as nn
import torchvision.models as models
from library.config import Config


class DualStatClassifier(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) Classifier designed for the Dual-Statistic Aggregation strategy.

    It accepts concatenated feature vectors (Mean Pooling + Max Pooling) representing a product,
    processes them through a regularized hidden layer, and outputs classification logits.

    Architecture:
        Input (4096) -> Linear(2048) -> BatchNorm -> ReLU -> Dropout -> Linear(Num_Classes)
    """

    def __init__(
        self,
        input_dim=Config.INPUT_DIM,
        num_classes=Config.NUM_CLASSES,
        dropout_rate=Config.DROPOUT_RATE,
    ):
        """
        Args:
            input_dim (int): Size of input features (default: 4096 for ResNet50 Mean+Max).
            num_classes (int): Number of target categories (default: 5270).
            dropout_rate (float): Probability of dropout (default: 0.25).
        """
        super(DualStatClassifier, self).__init__()

        # Map the combined statistical features back to the semantic dimension of the backbone
        hidden_dim = input_dim // 2

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights using Kaiming Normal for ReLU layers and standard settings for BN/Linear.
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch_Size, input_dim).

        Returns:
            torch.Tensor: Logits of shape (Batch_Size, num_classes).
        """
        x = self.layer1(x)
        logits = self.classifier(x)
        return logits


class ImageEncoder(nn.Module):
    """
    Feature extraction backbone using ResNet-50.

    This module is used to generate the embeddings for images before aggregation.
    It loads a pre-trained ResNet-50 and removes the final classification head.
    """

    def __init__(self):
        super(ImageEncoder, self).__init__()

        # Load ResNet-50 pre-trained on ImageNet
        # Using the modern weights API
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)

        # Remove the final fully connected layer (fc)
        # ResNet50 structure ends with: ... -> avgpool -> flatten -> fc
        # We keep everything up to and including avgpool
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        self.output_dim = 2048

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Batch of images (Batch, 3, 224, 224).

        Returns:
            torch.Tensor: Batch of feature vectors (Batch, 2048).
        """
        x = self.backbone(x)  # Output: (Batch, 2048, 1, 1)
        x = torch.flatten(x, 1)  # Output: (Batch, 2048)
        return x

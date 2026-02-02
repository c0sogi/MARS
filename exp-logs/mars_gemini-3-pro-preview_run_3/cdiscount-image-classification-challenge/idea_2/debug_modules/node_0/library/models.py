import torch
import torch.nn as nn
from torchvision import models
from library.config import EMBEDDING_DIM, NUM_CLASSES


class FrozenResNet(nn.Module):
    """
    A ResNet-18 based feature extractor with frozen weights.
    The classification head is removed to output 512-dimensional embeddings.
    """

    def __init__(self):
        super(FrozenResNet, self).__init__()

        # Load ResNet18 with ImageNet weights
        # Using try-except to handle potential torchvision version differences,
        # though 0.23.0 supports the weights enum.
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            self.backbone = models.resnet18(weights=weights)
        except AttributeError:
            self.backbone = models.resnet18(pretrained=True)

        # Replace the fully connected layer with Identity to extract features
        # ResNet18's average pooling output is flattened to 512 dimensions before the fc layer
        self.backbone.fc = nn.Identity()

        # Freeze all parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Ensure backbone stays in eval mode (specifically for BatchNorm stability)
        # even if the parent container is in train mode.
        self.backbone.eval()
        return self.backbone(x)

    def train(self, mode=True):
        # Override train to prevent the backbone from switching to train mode.
        # We want to use the pre-trained BatchNorm statistics, not update them.
        super(FrozenResNet, self).train(False)


class ProductClassifier(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) for classification based on embeddings.
    Projects the aggregated feature embeddings (512 dim) to the class logits.
    """

    def __init__(
        self,
        input_dim=EMBEDDING_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=2048,
        dropout=0.5,
    ):
        super(ProductClassifier, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.network(x)

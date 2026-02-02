import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from library.config import Config


class ManifoldMixupResNet(nn.Module):
    """
    ResNet-34 architecture with Manifold Mixup regularization.

    This model allows mixing of features at the Input level, after Layer 1,
    or after Layer 2, forcing the network to learn more robust representations
    in the deep feature space.
    """

    def __init__(self, num_classes=19, pretrained=True):
        """
        Args:
            num_classes (int): Number of output classes (bird species).
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(ManifoldMixupResNet, self).__init__()

        # Load ResNet-34 Backbone
        # Handle different torchvision versions for weight loading
        try:
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            backbone = models.resnet34(weights=weights)
        except AttributeError:
            backbone = models.resnet34(pretrained=pretrained)

        # Decompose backbone into blocks to allow intermediate access
        # Initial block: Conv -> BN -> ReLU -> MaxPool
        self.initial_layers = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool

        # Classification Head
        # Simple Linear Layer as per "Lesson 00007" (Avoid complex heads)
        self.in_features = backbone.fc.in_features
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x, target=None, mixup=False, alpha=0.2):
        """
        Forward pass with optional Manifold Mixup.

        Args:
            x (torch.Tensor): Input batch of images (N, 3, H, W).
            target (torch.Tensor, optional): Target labels (N, C). Required if mixup is True.
            mixup (bool): Whether to apply Manifold Mixup.
            alpha (float): Parameter for Beta distribution sampling.

        Returns:
            If mixup=True: (logits, target_a, target_b, lam)
            If mixup=False: logits
        """
        if mixup and target is not None:
            # --- Manifold Mixup Logic ---

            # 1. Sample mixing coefficient lambda
            lam = np.random.beta(alpha, alpha)

            # 2. Select layer to mix at: 0=Input, 1=After Layer1, 2=After Layer2
            # These correspond to the "Eligible Layers" in the strategy
            layer_k = np.random.choice([0, 1, 2])

            # 3. Generate permutation for mixing
            batch_size = x.size(0)
            index = torch.randperm(batch_size).to(x.device)

            target_a = target
            target_b = target[index]

            # 4. Sequential Processing with Injection
            out = x

            # Mix at Input (k=0)
            if layer_k == 0:
                out = lam * out + (1 - lam) * out[index]

            out = self.initial_layers(out)
            out = self.layer1(out)

            # Mix after Layer 1 (k=1)
            if layer_k == 1:
                out = lam * out + (1 - lam) * out[index]

            out = self.layer2(out)

            # Mix after Layer 2 (k=2)
            if layer_k == 2:
                out = lam * out + (1 - lam) * out[index]

            out = self.layer3(out)
            out = self.layer4(out)

            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            logits = self.fc(out)

            return logits, target_a, target_b, lam

        else:
            # --- Standard Forward Pass ---
            out = self.initial_layers(x)
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = self.avgpool(out)
            out = torch.flatten(out, 1)
            logits = self.fc(out)

            return logits

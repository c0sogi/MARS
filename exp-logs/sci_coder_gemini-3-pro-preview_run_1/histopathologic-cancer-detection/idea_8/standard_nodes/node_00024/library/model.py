import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import DenseNet121_Weights, DenseNet169_Weights


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.

    Computes the generalized mean: f(X) = (1/|X| * sum(x^p))^(1/p)

    Args:
        p (float): Initial value for the power parameter.
        eps (float): Small constant for numerical stability.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        # Apply clamping to avoid NaNs with negative inputs (though DenseNet features are usually ReLU'd)
        # or zeros when p < 1.
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


class EnsembleDenseNet(nn.Module):
    """
    Custom DenseNet architecture for Tumor Detection.

    Modifications:
    1. Input Stem: 7x7 stride-2 conv -> 3x3 stride-1 conv. MaxPool removed.
       Preserves 48x48 spatial resolution.
    2. Pooling: Global Average Pooling -> GeM Pooling.
    3. Classifier: Adjusted for binary classification.
    """

    def __init__(self, arch_name="densenet121", pretrained=True, num_classes=1):
        super(EnsembleDenseNet, self).__init__()

        # 1. Load Backbone
        if arch_name == "densenet121":
            weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.densenet121(weights=weights)
        elif arch_name == "densenet169":
            weights = DenseNet169_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.densenet169(weights=weights)
        else:
            raise ValueError(
                f"Architecture {arch_name} not supported. Choose 'densenet121' or 'densenet169'."
            )

        self.features = base_model.features

        # 2. Modify Input Stem
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # New:      Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        # We keep the original number of filters (64) to match the subsequent DenseBlock input.
        original_conv = self.features.conv0
        self.features.conv0 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )

        # Initialize the new conv layer
        # He initialization is generally good for ReLUs, but here we might want to preserve
        # some distribution properties or just use standard Kaiming.
        nn.init.kaiming_normal_(
            self.features.conv0.weight, mode="fan_out", nonlinearity="relu"
        )

        # Remove MaxPool to preserve spatial resolution
        # Original: MaxPool2d(kernel_size=3, stride=2, padding=1)
        # We replace it with Identity
        self.features.pool0 = nn.Identity()

        # 3. GeM Pooling
        self.gem_pool = GeM()

        # 4. Classifier
        # Get input features from the original classifier
        in_features = base_model.classifier.in_features
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        features = self.features(x)

        # DenseNet features usually end with ReLU, but let's ensure it before pooling
        out = F.relu(features, inplace=True)

        # GeM Pooling (replaces AdaptiveAvgPool)
        out = self.gem_pool(out)

        # Flatten
        out = torch.flatten(out, 1)

        # Classifier
        out = self.classifier(out)

        # We return logits. Sigmoid is applied in Loss (BCEWithLogits) or Inference.
        return out

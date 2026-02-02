import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the input tensor.
    Formula: f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: (B, C, H, W)
        # Clamp to avoid numerical instability with power operation
        x = x.clamp(min=eps)

        # Calculate x^p
        x = x.pow(p)

        # Average pooling over spatial dimensions (H, W)
        x = F.avg_pool2d(x, (x.size(-2), x.size(-1)))

        # Take the p-th root
        x = x.pow(1.0 / p)
        return x

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


class AppleResNet(nn.Module):
    """
    ResNet34 architecture with GeM Pooling and Multi-Sample Dropout.
    """

    def __init__(self):
        super(AppleResNet, self).__init__()

        # 1. Backbone
        # Initialize ResNet34 from timm
        # num_classes=0 and global_pool='' removes the default head and pooling
        # so we get the raw feature maps (B, C, H, W)
        self.backbone = timm.create_model(
            Config.BACKBONE, pretrained=Config.PRETRAINED, num_classes=0, global_pool=""
        )

        # Get the number of input features for the head (e.g., 512 for ResNet34)
        self.in_features = self.backbone.num_features

        # 2. Pooling Layer
        if Config.GEM_LEARNABLE:
            self.pooling = GeM(p=Config.GEM_P)
        else:
            # Fallback to standard Global Average Pooling
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # 3. Multi-Sample Dropout Head
        self.use_multi_sample = Config.USE_MULTI_SAMPLE_DROPOUT
        self.n_classes = Config.N_CLASSES

        if self.use_multi_sample:
            # Create a list of dropout layers with specified rates
            self.dropout_rates = Config.DROPOUT_RATES
            self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])
        else:
            # Standard single dropout
            self.dropouts = nn.ModuleList([nn.Dropout(0.5)])

        # Shared Fully Connected Layer
        # All dropout paths feed into this single shared layer
        self.fc = nn.Linear(self.in_features, self.n_classes)

        # Initialize weights for the FC layer
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input images of shape (B, C, H, W)

        Returns:
            torch.Tensor: Logits of shape (B, N_Classes).
                          If Multi-Sample Dropout is used, returns the average of logits.
        """
        # Feature Extraction
        # Output: (B, C, H, W)
        features = self.backbone(x)

        # Pooling
        # Output: (B, C, 1, 1)
        features = self.pooling(features)

        # Flatten
        # Output: (B, C)
        features = features.flatten(1)

        # Multi-Sample Dropout & Classification
        if self.use_multi_sample:
            logits = []
            for dropout in self.dropouts:
                # Apply specific dropout mask
                dropped_features = dropout(features)
                # Pass through shared FC layer
                out = self.fc(dropped_features)
                logits.append(out)

            # Stack logits: (B, N_Heads, N_Classes)
            stacked_logits = torch.stack(logits, dim=1)

            # Return the mean of logits (Internal Ensembling)
            # Output: (B, N_Classes)
            return torch.mean(stacked_logits, dim=1)

        else:
            # Standard forward pass
            features = self.dropouts[0](features)
            output = self.fc(features)
            return output

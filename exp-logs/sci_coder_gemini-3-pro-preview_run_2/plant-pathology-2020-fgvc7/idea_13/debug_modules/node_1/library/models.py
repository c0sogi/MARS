import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of the spatial features, focusing on high-activation regions.
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (Batch, Channels, Height, Width)
        # 1. Clamp to avoid numerical instability (NaNs) with power
        # 2. Raise to power p
        # 3. Average pool over spatial dimensions (H, W) -> (B, C, 1, 1)
        # 4. Raise to power 1/p
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


class AppleNet(nn.Module):
    """
    Heterogeneous Ensemble Component.
    Wraps a timm backbone with GeM Pooling and a Multi-Sample Dropout classification head.
    Outputs 2 logits corresponding to the decomposed binary targets (Rust, Scab).
    """

    def __init__(self, model_name, pretrained=True, dropout_rates=None):
        """
        Args:
            model_name (str): Name of the timm backbone (e.g., 'tf_efficientnetv2_l').
            pretrained (bool): Whether to load pretrained ImageNet weights.
            dropout_rates (list): List of dropout probabilities for Multi-Sample Dropout.
        """
        super(AppleNet, self).__init__()

        if dropout_rates is None:
            # Default to Config if not provided, though typically passed from Config
            dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.4]

        # 1. Load Backbone
        # num_classes=0 and global_pool='' ensures we get the raw spatial feature map (B, C, H, W)
        # instead of a pooled vector or logits.
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # 2. Determine Feature Dimension
        # timm models expose num_features
        self.in_features = self.backbone.num_features

        # 3. Generalized Mean Pooling
        # Replaces standard Global Average Pooling
        self.gem = GeM(p=Config.GEM_P)

        # 4. Multi-Sample Dropout Head
        self.dropout_rates = dropout_rates
        self.dropouts = nn.ModuleList([nn.Dropout(p) for p in self.dropout_rates])

        # Shared Linear Layer
        # Outputs logits for ['rust', 'scab']
        self.fc = nn.Linear(self.in_features, len(Config.TARGET_COLS))

        # Initialize weights
        nn.init.xavier_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        """
        Forward pass.
        Args:
            x (torch.Tensor): Input images (B, C, H, W)
        Returns:
            torch.Tensor: Logits (B, 2)
        """
        # Extract features from backbone: (B, C, H, W)
        x = self.backbone(x)

        # Apply GeM Pooling: (B, C, 1, 1)
        x = self.gem(x)

        # Flatten: (B, C)
        x = x.flatten(1)

        # Multi-Sample Dropout
        if len(self.dropouts) > 0:
            logits_list = []
            for dropout in self.dropouts:
                # Apply specific dropout mask
                out = dropout(x)
                # Pass through shared classifier
                out = self.fc(out)
                logits_list.append(out)

            # Average the logits from all dropout branches
            # This accelerates convergence and improves generalization
            logits = torch.stack(logits_list).mean(dim=0)
        else:
            # Fallback for no dropout
            logits = self.fc(x)

        return logits

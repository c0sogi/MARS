import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean: (1/N * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        # p is a learnable parameter initialized to 3.0
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: (Batch, Channels, Height, Width)
        # Clamp to avoid NaN and ensure non-negative base for power operation
        x = x.clamp(min=self.eps)

        # Calculate x^p
        x_pow = x.pow(self.p)

        # Average pooling over spatial dimensions (H, W)
        # Result shape: (Batch, Channels, 1, 1)
        avg_x_pow = F.avg_pool2d(x_pow, (x.size(-2), x.size(-1)))

        # Raise to power 1/p
        return avg_x_pow.pow(1.0 / self.p)


class CatheterModel(nn.Module):
    """
    Catheter Detection Model using ConvNeXt V2 Tiny backbone,
    GeM Pooling, and Multi-Sample Dropout.
    """

    def __init__(self, pretrained=True):
        super(CatheterModel, self).__init__()
        self.config = Config

        # Load Backbone
        # global_pool='' ensures we get spatial features (B, C, H, W) required for GeM
        # num_classes=0 removes the default classification head
        self.backbone = timm.create_model(
            self.config.backbone,
            pretrained=pretrained,
            in_chans=self.config.in_chans,
            num_classes=0,
            global_pool="",
        )

        # Determine the number of input features for the head
        # We run a dummy forward pass to be robust against different backbone variants
        with torch.no_grad():
            dummy_input = torch.randn(1, self.config.in_chans, 224, 224)
            features = self.backbone(dummy_input)
            self.num_features = features.shape[1]

        # Pooling Layer
        if self.config.use_gem_pooling:
            self.pooling = GeM()
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)

        # Multi-Sample Dropout (MSD)
        # We create a list of dropout layers to be applied in parallel
        if self.config.use_multi_sample_dropout:
            self.dropouts = nn.ModuleList(
                [nn.Dropout(self.config.msd_rate) for _ in range(self.config.msd_num)]
            )
        else:
            # Fallback to a single path with no dropout (or standard dropout if rate > 0)
            self.dropouts = nn.ModuleList([nn.Dropout(0.0)])

        # Final Classification Layer
        self.fc = nn.Linear(self.num_features, self.config.num_classes)

    def forward(self, x):
        # 1. Backbone Feature Extraction
        # Output shape: (B, C, H, W)
        x = self.backbone(x)

        # 2. Pooling
        # Output shape: (B, C, 1, 1)
        x = self.pooling(x)

        # Flatten: (B, C)
        x = x.flatten(1)

        # 3. Classification Head with Multi-Sample Dropout
        if self.config.use_multi_sample_dropout and self.training:
            # During training, apply multiple dropout masks and average the predictions
            logits_list = []
            for dropout in self.dropouts:
                # Apply dropout, then linear projection
                logits_list.append(self.fc(dropout(x)))

            # Stack and average logits
            # Shape: (B, num_classes)
            logits = torch.mean(torch.stack(logits_list), dim=0)
        else:
            # During inference (or if MSD is disabled), use a single path
            # self.dropouts[0] handles the scaling automatically in eval mode
            logits = self.fc(self.dropouts[0](x))

        return logits

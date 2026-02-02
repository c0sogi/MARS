import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Formula: f(x) = (1/N * sum(x^p))^(1/p)

    This pooling method is learnable (via parameter p) and allows the network
    to focus on salient features (high activations) more effectively than
    standard Average Pooling, which is crucial for detecting small disease spots.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x shape: [B, C, H, W]
        # Clamp input to avoid numerical instability (NaNs) with pow()
        x = x.clamp(min=self.eps)

        # Apply GeM formula
        # F.avg_pool2d calculates (1/N * sum(x^p)) over the spatial dimensions
        x_p = x.pow(self.p)
        x_p_avg = F.avg_pool2d(x_p, (x.size(-2), x.size(-1)))

        return x_p_avg.pow(1.0 / self.p)


class AppleDiseaseModel(nn.Module):
    """
    Apple Disease Detection Model.

    Architecture:
    1. Backbone (EfficientNetV2-L or ConvNeXt-Base) initialized via timm.
    2. GeM Pooling Layer.
    3. Multi-Sample Dropout Head (Internal Ensemble).
    4. Linear Classification Layer (Output: 2 classes [Rust, Scab]).
    """

    def __init__(
        self,
        model_name: str,
        pretrained: bool = True,
        num_classes: int = 2,
        gem_p: float = 3.0,
        num_msd: int = 5,
        msd_dropout: float = 0.2,
    ):
        super(AppleDiseaseModel, self).__init__()

        # 1. Load Backbone
        # num_classes=0 and global_pool='' ensures we get the raw spatial feature map [B, C, H, W]
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool=""
        )

        # Determine input features dimension
        if hasattr(self.backbone, "num_features"):
            self.in_features = self.backbone.num_features
        else:
            # Fallback: Infer shape via dummy pass
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                features = self.backbone(dummy)
                self.in_features = features.shape[1]

        # 2. GeM Pooling
        self.gem = GeM(p=gem_p)

        # 3. Multi-Sample Dropout
        self.num_msd = num_msd
        self.dropouts = nn.ModuleList([nn.Dropout(msd_dropout) for _ in range(num_msd)])

        # 4. Classifier
        self.fc = nn.Linear(self.in_features, num_classes)

    def forward(self, x):
        # Extract features from backbone
        x = self.backbone(x)  # [B, C, H, W]

        # Apply GeM Pooling
        x = self.gem(x)  # [B, C, 1, 1]

        # Flatten
        x = x.flatten(1)  # [B, C]

        if self.training and self.num_msd > 0:
            # Multi-Sample Dropout Strategy:
            # Pass features through multiple dropout masks, compute logits for each,
            # and average the results. This acts as an internal ensemble.
            logits_list = []
            for dropout_layer in self.dropouts:
                logits_list.append(self.fc(dropout_layer(x)))

            # Stack and average logits
            logits = torch.stack(logits_list, dim=0).mean(dim=0)
        else:
            # Inference Mode:
            # Standard forward pass. Dropout layers are identity in eval mode.
            logits = self.fc(x)

        return logits

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    Computes the generalized mean of each channel in the input tensor.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x: (B, C, H, W)
        # Global pooling over spatial dimensions (H, W)
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


class AppleDiseaseFPN(nn.Module):
    """
    Feature Pyramid Network with Deep Supervision for Apple Disease Detection.

    Architecture:
    1. Backbone (timm): Extracts multi-scale features (C3, C4, C5).
    2. FPN: Top-down pathway fusing semantic context (C5) into high-res features (C3).
    3. Heads: Independent classification heads on each pyramid level (P3, P4, P5) using GeM pooling.
    """

    def __init__(self, model_name, num_classes=4, pretrained=True, fpn_dim=256):
        """
        Args:
            model_name (str): Name of the timm backbone.
            num_classes (int): Number of target classes.
            pretrained (bool): Whether to load pretrained backbone weights.
            fpn_dim (int): Channel dimension for the FPN lateral connections.
        """
        super(AppleDiseaseFPN, self).__init__()

        # 1. Backbone
        # Use features_only=True to get intermediate feature maps.
        # We target the last 3 stages, typically strides 8, 16, 32.
        # Using indices (-3, -2, -1) is a generic way to get the last three feature maps.
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            # out_indices removed to return all stages, then we slice the last 3
        )

        # Dynamically determine feature channels
        # We run a dummy pass to get the channel dimensions of C3, C4, C5
        dummy_size = 256  # Sufficient size to pass through stride 32
        dummy_in = torch.randn(2, 3, dummy_size, dummy_size)
        with torch.no_grad():
            features = self.backbone(dummy_in)

        # features list order: [..., C3 (stride 8), C4 (stride 16), C5 (stride 32)]
        # We take the last 3 feature maps
        c3_channels = features[-3].shape[1]
        c4_channels = features[-2].shape[1]
        c5_channels = features[-1].shape[1]

        # 2. FPN Lateral Connections (1x1 convs)
        # Project all backbone features to a common FPN dimension
        self.lat_c5 = nn.Conv2d(c5_channels, fpn_dim, kernel_size=1, bias=False)
        self.lat_c4 = nn.Conv2d(c4_channels, fpn_dim, kernel_size=1, bias=False)
        self.lat_c3 = nn.Conv2d(c3_channels, fpn_dim, kernel_size=1, bias=False)

        # Batch Normalization for stability after projection
        self.bn_c5 = nn.BatchNorm2d(fpn_dim)
        self.bn_c4 = nn.BatchNorm2d(fpn_dim)
        self.bn_c3 = nn.BatchNorm2d(fpn_dim)

        # 3. Classification Heads
        # Attach a head to each pyramid level for Deep Supervision
        # P3: Finest resolution, fused semantics + details (Main Output)
        # P4, P5: Coarser resolutions (Auxiliary Outputs)
        self.head_p5 = self._build_head(fpn_dim, num_classes)
        self.head_p4 = self._build_head(fpn_dim, num_classes)
        self.head_p3 = self._build_head(fpn_dim, num_classes)

    def _build_head(self, in_channels, num_classes):
        """
        Constructs a classification head with GeM pooling.
        """
        return nn.Sequential(
            GeM(),
            nn.Flatten(),
            nn.Linear(in_channels, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),  # Swish activation
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        # 1. Backbone Feature Extraction
        # feats: [..., C3, C4, C5]
        feats = self.backbone(x)
        c3, c4, c5 = feats[-3], feats[-2], feats[-1]

        # 2. FPN Top-Down Pathway

        # Level 5 (Coarsest)
        p5 = self.bn_c5(self.lat_c5(c5))

        # Level 4: Project C4 + Upsample P5
        # Use nearest neighbor upsampling for simplicity and efficiency
        p5_up = F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p4 = self.bn_c4(self.lat_c4(c4)) + p5_up

        # Level 3 (Finest): Project C3 + Upsample P4
        p4_up = F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        p3 = self.bn_c3(self.lat_c3(c3)) + p4_up

        # 3. Prediction Heads
        logits_p5 = self.head_p5(p5)
        logits_p4 = self.head_p4(p4)
        logits_p3 = self.head_p3(p3)

        if self.training:
            # Deep Supervision: Return all logits for loss calculation
            # Loss = L(p3) + 0.5 * (L(p4) + L(p5))
            return logits_p3, logits_p4, logits_p5
        else:
            # Inference
            # Return the finest scale prediction (P3)
            # P3 contains the most detailed spatial information fused with global context
            return logits_p3

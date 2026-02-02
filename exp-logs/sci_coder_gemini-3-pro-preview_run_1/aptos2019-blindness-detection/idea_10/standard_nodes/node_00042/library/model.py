import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from library.config import Config


class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) layer.
    Computes the generalized mean of the input tensor.
    f(X) = (1/|X| * sum(x^p))^(1/p)
    """

    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        # x shape: [B, C, H, W]
        # Clamp for numerical stability
        x = x.clamp(min=eps)
        # Average pooling on x^p
        # Kernel size is the spatial size of x (H, W) -> Global Pooling
        return F.avg_pool2d(x.pow(p), (x.size(-2), x.size(-1))).pow(1.0 / p)

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


class MultiScaleConvNeXt(nn.Module):
    """
    Multi-Scale ConvNeXt Architecture for Diabetic Retinopathy Detection.

    Features:
    1. Backbone: ConvNeXt Base (ImageNet-22k pretrained).
    2. Multi-Scale Extraction: Uses features from Stage 3 (1/16) and Stage 4 (1/32).
    3. Dual-Stream Pooling: Applies both GAP and GeM to both stages.
    4. Head: LayerNorm -> Multi-Sample Dropout -> Ordinal Linear Head.
    """

    def __init__(self, pretrained=True):
        super(MultiScaleConvNeXt, self).__init__()

        # 1. Backbone
        # features_only=True allows extraction of intermediate layers
        # out_indices=(2, 3) extracts the last two stages (Stage 3 and Stage 4)
        self.backbone = timm.create_model(
            Config.backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3),
            drop_path_rate=Config.drop_path_rate,
        )

        # Get channel counts for the extracted stages
        # feature_info.channels() returns a list of channel counts for the selected indices
        feature_channels = self.backbone.feature_info.channels()
        self.c3 = feature_channels[0]  # Channels for Stage 3
        self.c4 = feature_channels[1]  # Channels for Stage 4

        # 2. Pooling Layers
        self.gem_pool = GeM()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 3. Calculate Embedding Dimension
        # We concatenate GAP and GeM for both stages: (C3*2) + (C4*2)
        self.embedding_dim = (self.c3 * 2) + (self.c4 * 2)

        # 4. Classification Head
        self.neck_ln = nn.LayerNorm(self.embedding_dim)

        # Multi-Sample Dropout: Improves generalization by averaging gradients from multiple dropout masks
        self.num_dropouts = 5
        self.dropouts = nn.ModuleList(
            [nn.Dropout(Config.head_dropout) for _ in range(self.num_dropouts)]
        )

        # Linear Layer for Ordinal Regression
        # Outputs 4 logits corresponding to the thresholds: 0->1, 1->2, 2->3, 3->4
        self.fc = nn.Linear(self.embedding_dim, Config.num_ordinal_heads)

        # Initialize weights for the head
        self._init_weights(self.fc)
        self._init_weights(self.neck_ln)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.weight, 1.0)
            nn.init.constant_(module.bias, 0)

    def forward(self, x):
        # Backbone Forward Pass
        # Returns a list of tensors: [Stage3_FeatureMap, Stage4_FeatureMap]
        features = self.backbone(x)
        f3 = features[0]  # Shape: [B, C3, H/16, W/16]
        f4 = features[1]  # Shape: [B, C4, H/32, W/32]

        # Stage 3 Pooling
        f3_avg = self.avg_pool(f3).flatten(1)
        f3_gem = self.gem_pool(f3).flatten(1)

        # Stage 4 Pooling
        f4_avg = self.avg_pool(f4).flatten(1)
        f4_gem = self.gem_pool(f4).flatten(1)

        # Concatenate all features
        # Vector contains: [Global Context (S4), Local Texture (S3), Dominant Features (GeM), Average Features (GAP)]
        combined_features = torch.cat([f3_avg, f3_gem, f4_avg, f4_gem], dim=1)

        # Normalization
        combined_features = self.neck_ln(combined_features)

        # Multi-Sample Dropout & Classification
        # Pass the features through multiple dropout masks and average the predictions
        logits_list = []
        for dropout in self.dropouts:
            logits_list.append(self.fc(dropout(combined_features)))

        # Average the logits
        logits = torch.stack(logits_list, dim=0).mean(dim=0)

        return logits

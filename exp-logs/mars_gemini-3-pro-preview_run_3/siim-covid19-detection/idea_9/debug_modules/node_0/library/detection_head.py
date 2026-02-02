import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from library.config import Config


class ATSSHead(nn.Module):
    """
    ATSS Detection Head with Anchor Generator.
    Produces classification logits and regression offsets for each FPN level.
    """

    def __init__(self, in_channels, num_classes, num_anchors=1):
        super(ATSSHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared towers for classification and regression
        # 4 convolutions with GroupNorm and ReLU
        self.cls_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )
        self.reg_tower = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True),
        )

        # Prediction heads
        self.cls_pred = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg_pred = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)

        # Learnable scale parameter per FPN level (P3, P4, P5)
        self.scales = nn.ModuleList([nn.Parameter(torch.ones(1)) for _ in range(3)])

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias initialization for classification to stabilize training (Focal Loss)
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def _generate_anchors(self, feature_maps, strides):
        """
        Generates anchors for each feature map level.

        Args:
            feature_maps: List of tensors [P3, P4, P5]
            strides: List of strides [8, 16, 32]

        Returns:
            anchors_all: Tensor [N_all, 4] (cx, cy, stride, stride)
            num_anchors_per_level: List of integers
        """
        anchors_all = []
        num_anchors_per_level = []

        for feat, stride in zip(feature_maps, strides):
            B, C, H, W = feat.shape
            device = feat.device

            # Grid of centers
            y = torch.arange(H, device=device) * stride + stride // 2
            x = torch.arange(W, device=device) * stride + stride // 2

            # Meshgrid (indexing='ij' for y, x order)
            gy, gx = torch.meshgrid(y, x, indexing="ij")

            # Stack to get (H, W, 2) -> (cx, cy)
            centers = torch.stack([gx, gy], dim=-1)

            # Use stride as the base size for the anchor (ATSS logic uses this)
            sizes = torch.full_like(centers, stride)

            # Concatenate to get (cx, cy, stride, stride)
            level_anchors = torch.cat([centers, sizes], dim=-1).view(-1, 4)

            anchors_all.append(level_anchors)
            num_anchors_per_level.append(len(level_anchors))

        return torch.cat(anchors_all, dim=0), num_anchors_per_level

    def forward(self, feats):
        """
        Args:
            feats: List of feature maps [P3, P4, P5]

        Returns:
            cls_logits: [B, N_all, num_classes]
            bbox_preds: [B, N_all, 4]
            anchors: [N_all, 4]
            num_anchors_list: List of ints
        """
        # Strides corresponding to P3, P4, P5
        strides = [8, 16, 32]

        cls_logits_all = []
        bbox_preds_all = []

        for i, x in enumerate(feats):
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)

            # Classification prediction
            cls_out = self.cls_pred(cls_feat)
            # Reshape: (B, A*C, H, W) -> (B, H, W, A*C) -> (B, N, C)
            B, _, H, W = cls_out.shape
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            cls_logits_all.append(cls_out)

            # Regression prediction
            reg_out = self.reg_pred(reg_feat)
            # Apply learnable scale
            reg_out = reg_out * self.scales[i]
            # Reshape: (B, A*4, H, W) -> (B, N, 4)
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 4)
            # Exponential to ensure positive values
            reg_out = torch.exp(reg_out)
            bbox_preds_all.append(reg_out)

        # Concatenate predictions from all levels
        cls_logits = torch.cat(cls_logits_all, dim=1)
        bbox_preds = torch.cat(bbox_preds_all, dim=1)

        # Generate anchors
        anchors, num_anchors_list = self._generate_anchors(feats, strides)

        return cls_logits, bbox_preds, anchors, num_anchors_list


class QueryClassifier(nn.Module):
    """
    Query-Based Global Classifier for Study Labels.
    Uses learnable 'Diagnosis Queries' to attend to multi-scale features via a Transformer Decoder.
    """

    def __init__(self, in_channels, num_classes=4):
        super(QueryClassifier, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = in_channels

        # Learnable Queries: (1, NumClasses, EmbedDim)
        # Each query represents a specific diagnosis (Negative, Typical, Indeterminate, Atypical)
        self.queries = nn.Parameter(torch.randn(1, num_classes, in_channels))

        # Transformer Decoder Layer components
        self.mha = nn.MultiheadAttention(
            embed_dim=in_channels, num_heads=8, batch_first=True
        )
        self.norm = nn.LayerNorm(in_channels)

        self.ffn = nn.Sequential(
            nn.Linear(in_channels, in_channels * 4),
            nn.ReLU(),
            nn.Linear(in_channels * 4, in_channels),
        )
        self.norm2 = nn.LayerNorm(in_channels)

        # Final projection to logits (1 scalar per query)
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, feats):
        """
        Args:
            feats: List of feature maps [P3, P4, P5]

        Returns:
            logits: [B, num_classes]
        """
        B = feats[0].shape[0]

        # Flatten and concatenate all feature maps to form the memory sequence
        flattened = []
        for f in feats:
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            flattened.append(f.flatten(2).transpose(1, 2))

        memory = torch.cat(flattened, dim=1)

        # Expand queries to match batch size: (B, NumClasses, C)
        queries = self.queries.expand(B, -1, -1)

        # Multi-Head Attention
        # Queries attend to Memory (Image Features)
        attn_out, _ = self.mha(queries, memory, memory)
        queries = self.norm(queries + attn_out)

        # Feed Forward Network
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        # Classification Projection
        # (B, NumClasses, C) -> (B, NumClasses, 1) -> (B, NumClasses)
        logits = self.classifier(queries).squeeze(-1)

        return logits

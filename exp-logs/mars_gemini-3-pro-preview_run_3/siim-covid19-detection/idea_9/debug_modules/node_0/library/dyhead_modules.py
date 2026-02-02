import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import timm
from library.config import Config


class SwinBackbone(nn.Module):
    """
    Swin Transformer Backbone using timm.
    Extracts features from stages with strides 8, 16, and 32.
    """

    def __init__(self):
        super(SwinBackbone, self).__init__()
        # Swin Tiny: Stages 0 (4x), 1 (8x), 2 (16x), 3 (32x)
        # We want strides 8, 16, 32 -> indices 1, 2, 3
        self.model = timm.create_model(
            Config.BACKBONE,
            pretrained=True,
            features_only=True,
            out_indices=(1, 2, 3),
        )
        self.out_channels = Config.BACKBONE_OUT_CHANNELS  # [192, 384, 768]

    def forward(self, x):
        return self.model(x)


class ScaleAwareAttention(nn.Module):
    """
    DyHead Scale-aware Attention.
    Fuses features across different levels (L).
    """

    def __init__(self, channels):
        super(ScaleAwareAttention, self).__init__()
        # Input shape: (B, C, L, H, W)
        # We mix L. Using a simple linear layer shared across channels.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels, bias=False),
            nn.Hardsigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L, H, W)
        B, C, L, H, W = x.shape
        # Pool spatial dims -> (B, C, L, 1, 1)
        out = F.avg_pool3d(x, kernel_size=(1, H, W))
        out = out.view(B, C, L)
        # We want to learn weights for L based on C?
        # Paper: sigma(f(mean(x)))
        # Simplified implementation: Mix C to get weights for L?
        # Or mix L? The paper says "dynamically fuses features across different scales".
        # Let's implement channel-wise scale attention.
        # (B, C, L) -> Linear -> (B, C, L)
        # We treat L as a spatial dim effectively for the FC? No.
        # Let's permute to apply Linear on C.
        # Actually, standard DyHead implementation:
        # 1. AvgPool spatial
        # 2. Linear(C -> 1) -> Sigmoid? No.
        # 3. Paper formula: pi_scale(F) * F.
        # Let's use a robust implementation:
        # Learn a weight vector of shape (B, C, L) derived from (B, C, L)
        out = out.permute(0, 2, 1)  # (B, L, C)
        out = self.fc(out)  # (B, L, C)
        out = out.permute(0, 2, 1).view(B, C, L, 1, 1)
        return x * out


class SpatialAwareAttention(nn.Module):
    """
    DyHead Spatial-aware Attention.
    Uses Deformable Convolution v2.
    """

    def __init__(self, channels):
        super(SpatialAwareAttention, self).__init__()
        self.channels = channels
        # 3x3 DeformConv
        self.kernel_size = 3
        self.padding = 1

        # Offset generator: 2*k*k offsets + k*k masks = 18 + 9 = 27
        self.offset_conv = nn.Conv2d(
            channels, 3 * self.kernel_size**2, kernel_size=3, padding=1
        )

        # The actual convolution weight (acting as the attention mechanism)
        # We keep channels same, so groups=channels (depthwise) is often used for attention,
        # but DyHead usually does full conv. Let's stick to full conv C->C.
        self.conv_weight = nn.Parameter(
            torch.empty(channels, channels, self.kernel_size, self.kernel_size)
        )
        self.conv_bias = nn.Parameter(torch.empty(channels))

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.conv_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv_weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.conv_bias, -bound, bound)

        # Initialize offset conv to 0 for stability
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x):
        # x: (B, C, L, H, W)
        B, C, L, H, W = x.shape

        # Collapse B and L for spatial processing
        x_reshaped = x.view(B * L, C, H, W)

        # Generate offsets and masks
        out = self.offset_conv(x_reshaped)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        # Apply DeformConv
        # torchvision.ops.deform_conv2d(input, offset, weight, bias, stride, padding, dilation, mask)
        out = torchvision.ops.deform_conv2d(
            x_reshaped,
            offset,
            self.conv_weight,
            self.conv_bias,
            stride=1,
            padding=self.padding,
            dilation=1,
            mask=mask,
        )

        return out.view(B, C, L, H, W)


class TaskAwareAttention(nn.Module):
    """
    DyHead Task-aware Attention (Dynamic ReLU).
    """

    def __init__(self, channels):
        super(TaskAwareAttention, self).__init__()
        # Predict 4 coefficients per channel: alpha1, beta1, alpha2, beta2
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.LayerNorm(channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, channels * 4),
        )

    def forward(self, x):
        # x: (B, C, L, H, W)
        B, C, L, H, W = x.shape

        # Global Average Pool over spatial+level dims
        # Paper suggests pooling over spatial, keeping level? Or global?
        # Usually global for task switching.
        avg = F.avg_pool3d(x, kernel_size=(L, H, W)).view(B, C)

        # Generate coeffs
        ctrl = self.fc(avg).view(B, C, 4)

        # Normalize (usually implicit in DyHead via specific activation, but here we use raw)
        # Split
        a1 = ctrl[:, :, 0].view(B, C, 1, 1, 1) + 1.0  # Bias 1 for stability
        b1 = ctrl[:, :, 1].view(B, C, 1, 1, 1)
        a2 = ctrl[:, :, 2].view(B, C, 1, 1, 1)
        b2 = ctrl[:, :, 3].view(B, C, 1, 1, 1)

        # Dynamic ReLU: max(a1*x + b1, a2*x + b2)
        return torch.max(a1 * x + b1, a2 * x + b2)


class DyHeadBlock(nn.Module):
    def __init__(self, channels):
        super(DyHeadBlock, self).__init__()
        self.scale_attn = ScaleAwareAttention(channels)
        self.spatial_attn = SpatialAwareAttention(channels)
        self.task_attn = TaskAwareAttention(channels)

    def forward(self, x):
        # x: (B, C, L, H, W)
        out = self.scale_attn(x)
        out = self.spatial_attn(out)
        out = self.task_attn(out)
        return out


class DyHead(nn.Module):
    """
    Dynamic Head Neck.
    """

    def __init__(self, in_channels, out_channels, num_blocks=6):
        super(DyHead, self).__init__()
        self.out_channels = out_channels

        # Lateral convolutions to project to common channel dim
        self.lateral_convs = nn.ModuleList()
        for ch in in_channels:
            self.lateral_convs.append(nn.Conv2d(ch, out_channels, kernel_size=1))

        # DyHead Blocks
        self.blocks = nn.ModuleList(
            [DyHeadBlock(out_channels) for _ in range(num_blocks)]
        )

    def forward(self, inputs):
        # inputs: list of [P3, P4, P5] (strides 8, 16, 32)
        assert len(inputs) == 3

        # 1. Project to common channels
        feats = [conv(x) for conv, x in zip(self.lateral_convs, inputs)]

        # 2. Resize to median scale (P4, stride 16)
        # P3 (stride 8) -> Downsample
        # P5 (stride 32) -> Upsample
        median_feat = feats[1]
        target_h, target_w = median_feat.shape[2], median_feat.shape[3]

        resized_feats = []
        # P3 -> P4
        resized_feats.append(
            F.interpolate(
                feats[0],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        )
        # P4
        resized_feats.append(feats[1])
        # P5 -> P4
        resized_feats.append(
            F.interpolate(
                feats[2],
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        )

        # 3. Stack: (B, C, L, H, W)
        stacked_feats = torch.stack(resized_feats, dim=2)

        # 4. Apply Blocks
        for block in self.blocks:
            stacked_feats = block(stacked_feats)

        # 5. Unstack and resize back
        refined_feats = []

        # P3 (stride 8)
        p3_out = stacked_feats[:, :, 0, :, :]
        p3_h, p3_w = inputs[0].shape[2], inputs[0].shape[3]
        refined_feats.append(
            F.interpolate(
                p3_out, size=(p3_h, p3_w), mode="bilinear", align_corners=False
            )
        )

        # P4 (stride 16)
        refined_feats.append(stacked_feats[:, :, 1, :, :])

        # P5 (stride 32)
        p5_out = stacked_feats[:, :, 2, :, :]
        p5_h, p5_w = inputs[2].shape[2], inputs[2].shape[3]
        refined_feats.append(
            F.interpolate(
                p5_out, size=(p5_h, p5_w), mode="bilinear", align_corners=False
            )
        )

        return refined_feats


class ATSSHead(nn.Module):
    """
    ATSS Detection Head with Anchor Generator.
    """

    def __init__(self, in_channels, num_classes, num_anchors=1):
        super(ATSSHead, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Shared towers
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

        # Learnable scale per level
        self.scales = nn.ModuleList([nn.Parameter(torch.ones(1)) for _ in range(3)])

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Bias init for classification to prevent instability at start
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        nn.init.constant_(self.cls_pred.bias, bias_value)

    def _generate_anchors(self, feature_maps, strides):
        """
        Generate anchors for ATSS.
        Returns: (N_all, 4) in format (cx, cy, stride, stride)
        """
        anchors_all = []
        num_anchors_per_level = []

        for feat, stride in zip(feature_maps, strides):
            B, C, H, W = feat.shape
            device = feat.device

            # Grid of centers
            y = torch.arange(H, device=device) * stride + stride // 2
            x = torch.arange(W, device=device) * stride + stride // 2

            # Meshgrid
            gy, gx = torch.meshgrid(y, x, indexing="ij")

            # (H, W, 4) -> (cx, cy, stride, stride)
            # We store stride as width/height proxy for ATSS matching logic
            centers = torch.stack([gx, gy], dim=-1)  # (H, W, 2)
            sizes = torch.full_like(centers, stride)

            level_anchors = torch.cat([centers, sizes], dim=-1).view(-1, 4)
            anchors_all.append(level_anchors)
            num_anchors_per_level.append(len(level_anchors))

        return torch.cat(anchors_all, dim=0), num_anchors_per_level

    def forward(self, feats):
        # feats: [P3, P4, P5]
        strides = [8, 16, 32]

        cls_logits_all = []
        bbox_preds_all = []

        for i, x in enumerate(feats):
            cls_feat = self.cls_tower(x)
            reg_feat = self.reg_tower(x)

            # Classification
            cls_out = self.cls_pred(cls_feat)
            # (B, A*C, H, W) -> (B, H, W, A*C) -> (B, N, C)
            B, _, H, W = cls_out.shape
            cls_out = cls_out.permute(0, 2, 3, 1).reshape(B, -1, self.num_classes)
            cls_logits_all.append(cls_out)

            # Regression
            reg_out = self.reg_pred(reg_feat)
            # Apply scale
            reg_out = reg_out * self.scales[i]
            # (B, A*4, H, W) -> (B, N, 4)
            reg_out = reg_out.permute(0, 2, 3, 1).reshape(B, -1, 4)
            # Exp to ensure positive offsets
            reg_out = torch.exp(reg_out)
            bbox_preds_all.append(reg_out)

        cls_logits = torch.cat(cls_logits_all, dim=1)
        bbox_preds = torch.cat(bbox_preds_all, dim=1)

        anchors, num_anchors_list = self._generate_anchors(feats, strides)

        return cls_logits, bbox_preds, anchors, num_anchors_list


class QueryClassifier(nn.Module):
    """
    Query-Based Global Classifier for Study Labels.
    Uses learnable queries to attend to multi-scale features.
    """

    def __init__(self, in_channels, num_classes=4):
        super(QueryClassifier, self).__init__()
        self.num_classes = num_classes
        self.embed_dim = in_channels

        # Learnable Queries: (1, NumClasses, EmbedDim)
        # Each query corresponds to one study class
        self.queries = nn.Parameter(torch.randn(1, num_classes, in_channels))

        # Transformer Decoder Layer
        # We use a simplified attention mechanism
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

        # Final projection to logits
        # Since each query is class-specific, we project to 1 scalar per query
        self.classifier = nn.Linear(in_channels, 1)

    def forward(self, feats):
        # feats: list of [P3, P4, P5]
        B = feats[0].shape[0]

        # Flatten and concatenate all features: (B, SeqLen, C)
        flattened = []
        for f in feats:
            # (B, C, H, W) -> (B, C, H*W) -> (B, H*W, C)
            flattened.append(f.flatten(2).transpose(1, 2))

        memory = torch.cat(flattened, dim=1)

        # Expand queries to batch size
        queries = self.queries.expand(B, -1, -1)

        # Attention
        # Q=Queries, K=V=Memory
        attn_out, _ = self.mha(queries, memory, memory)
        queries = self.norm(queries + attn_out)

        # FFN
        ffn_out = self.ffn(queries)
        queries = self.norm2(queries + ffn_out)

        # Classifier
        # (B, 4, C) -> (B, 4, 1) -> (B, 4)
        logits = self.classifier(queries).squeeze(-1)

        return logits


class SwinDyHeadModel(nn.Module):
    """
    Unified Model: Swin + DyHead + ATSS + QueryClassifier.
    """

    def __init__(self):
        super(SwinDyHeadModel, self).__init__()

        # 1. Backbone
        self.backbone = SwinBackbone()

        # 2. Neck (DyHead)
        self.dyhead = DyHead(
            in_channels=self.backbone.out_channels,
            out_channels=Config.DYHEAD_CHANNELS,
            num_blocks=Config.DYHEAD_NUM_BLOCKS,
        )

        # 3. Detection Head (ATSS)
        self.det_head = ATSSHead(
            in_channels=Config.DYHEAD_CHANNELS, num_classes=Config.NUM_CLASSES_DET
        )

        # 4. Study Head (Query Classifier)
        self.study_head = QueryClassifier(
            in_channels=Config.DYHEAD_CHANNELS, num_classes=Config.NUM_CLASSES_STUDY
        )

    def forward(self, x):
        # x: (B, 3, H, W)

        # Backbone features
        features = self.backbone(x)

        # DyHead refinement
        features = self.dyhead(features)

        # Detection
        cls_logits, bbox_preds, anchors, num_anchors_list = self.det_head(features)

        # Study Classification
        study_logits = self.study_head(features)

        # Return dict for Criterion
        return {
            "cls_logits": cls_logits,
            "bbox_preds": bbox_preds,
            "anchors": anchors,
            "num_anchors_per_level": num_anchors_list,
            "study_logits": study_logits,
        }

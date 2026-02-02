import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align, nms

from library.config import Config
from library.backbone import build_backbone, NestedTensor


class MLP(nn.Module):
    """Simple Multi-Layer Perceptron"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ATSSHead(nn.Module):
    """
    Auxiliary One-Stage Detection Head (simplified ATSS/FCOS style).
    Operates on Encoder feature maps.
    """

    def __init__(self, hidden_dim, num_classes, num_anchors=1):
        super().__init__()
        self.cls_convs = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
        )
        self.reg_convs = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(32, hidden_dim),
            nn.ReLU(),
        )

        # Class prediction
        self.cls_logits = nn.Conv2d(hidden_dim, num_classes * num_anchors, 3, padding=1)
        # Box regression (l, t, r, b)
        self.bbox_pred = nn.Conv2d(hidden_dim, 4 * num_anchors, 3, padding=1)
        # Centerness
        self.centerness = nn.Conv2d(hidden_dim, 1 * num_anchors, 3, padding=1)

        # Scale parameter for regression
        self.scales = nn.ModuleList(
            [nn.Parameter(torch.ones(1)) for _ in range(5)]
        )  # Max 5 levels

    def forward(self, feature_maps):
        """
        Args:
            feature_maps: List of tensors [B, C, H, W]
        Returns:
            logits, bbox_reg, centerness
        """
        logits = []
        bbox_regs = []
        centerness = []

        for i, x in enumerate(feature_maps):
            cls_feat = self.cls_convs(x)
            reg_feat = self.reg_convs(x)

            logits.append(self.cls_logits(cls_feat))
            centerness.append(self.centerness(reg_feat))

            # Apply scale and exp for positive regression values
            scale = self.scales[min(i, len(self.scales) - 1)]
            bbox_reg = torch.exp(scale * self.bbox_pred(reg_feat))
            bbox_regs.append(bbox_reg)

        return logits, bbox_regs, centerness


class RCNNHead(nn.Module):
    """
    Auxiliary Two-Stage Detection Head.
    Uses ROI Align on Encoder features given proposals.
    """

    def __init__(self, hidden_dim, num_classes, roi_resolution=7):
        super().__init__()
        self.roi_resolution = roi_resolution

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * roi_resolution**2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.cls_score = nn.Linear(1024, num_classes + 1)  # +1 for background
        self.bbox_pred = nn.Linear(1024, 4)  # Class agnostic regression

    def forward(self, feature_maps, proposals, image_shapes):
        """
        Args:
            feature_maps: List of tensors [B, C, H, W] (Encoder outputs)
            proposals: List of Tensors [N, 4] (Boxes in absolute coords)
            image_shapes: List of (h, w)
        """
        # We map proposals to feature levels based on scale (simplified: use single level or average)
        # For simplicity in this implementation, we use the highest resolution map (index 0)
        # or a specific stride. Let's use the middle map (stride 16) if available.

        if len(feature_maps) > 1:
            x = feature_maps[1]  # Stride 16 usually
            spatial_scale = 1.0 / 16.0  # Approx
        else:
            x = feature_maps[0]
            spatial_scale = 1.0 / 32.0

        # ROI Align
        # proposals must be a list of tensors
        box_features = roi_align(
            x,
            proposals,
            output_size=(self.roi_resolution, self.roi_resolution),
            spatial_scale=spatial_scale,
            sampling_ratio=2,
        )

        # Flatten
        box_features = box_features.flatten(1)

        # Predict
        fc_out = self.fc(box_features)
        cls_logits = self.cls_score(fc_out)
        bbox_deltas = self.bbox_pred(fc_out)

        return cls_logits, bbox_deltas


class CoDETR(nn.Module):
    def __init__(self, config=Config):
        super().__init__()
        self.config = config

        # 1. Backbone
        self.backbone = build_backbone(config)

        # 2. Transformer Projection
        # Backbone returns features at strides 8, 16, 32. Channels vary.
        # We project all to hidden_dim
        self.input_proj = nn.ModuleList(
            [
                nn.Conv2d(c, config.HIDDEN_DIM, kernel_size=1)
                for c in self.backbone.num_channels
            ]
        )

        # 3. Transformer
        self.transformer = nn.Transformer(
            d_model=config.HIDDEN_DIM,
            nhead=config.NHEADS,
            num_encoder_layers=config.NUM_ENCODER_LAYERS,
            num_decoder_layers=config.NUM_DECODER_LAYERS,
            dim_feedforward=config.DIM_FEEDFORWARD,
            dropout=config.DROPOUT,
            activation="relu",
            normalize_before=False,
        )

        # 4. Queries
        # Object Queries
        self.query_embed = nn.Embedding(config.NUM_QUERIES, config.HIDDEN_DIM)
        # Diagnosis Query (1 query for study level)
        self.diagnosis_query_embed = nn.Embedding(1, config.HIDDEN_DIM)

        # 5. Prediction Heads (Main Decoder)
        # Class Embed: +1 for background ("no object")
        self.class_embed = nn.Linear(config.HIDDEN_DIM, config.NUM_CLASSES + 1)
        self.bbox_embed = MLP(config.HIDDEN_DIM, config.HIDDEN_DIM, 4, 3)
        self.study_embed = nn.Linear(config.HIDDEN_DIM, config.NUM_STUDY_CLASSES)

        # 6. Auxiliary Heads (Encoder Supervision)
        self.aux_atss = ATSSHead(config.HIDDEN_DIM, config.NUM_CLASSES)
        self.aux_rcnn = RCNNHead(config.HIDDEN_DIM, config.NUM_CLASSES)

        # Level Embeddings for multi-scale transformer
        self.level_embed = nn.Parameter(torch.Tensor(3, config.HIDDEN_DIM))
        nn.init.normal_(self.level_embed)

    def forward(self, samples):
        """
        Args:
            samples: NestedTensor (tensors, mask)
                     tensors: [B, 3, H, W]
                     mask: [B, H, W]
        """
        # --- 1. Backbone & Projection ---
        features, pos = self.backbone(samples)

        srcs = []
        masks = []
        pos_embeds = []

        # Prepare multi-scale inputs
        for l, (src, pos_emb) in enumerate(zip(features, pos)):
            # src is NestedTensor
            src_tensor, src_mask = src.decompose()

            # Project to hidden_dim
            src_tensor = self.input_proj[l](src_tensor)

            srcs.append(src_tensor)
            masks.append(src_mask)
            pos_embeds.append(pos_emb)

        # Flatten and concatenate for Transformer
        # src: [B, C, H, W] -> [B, C, HW] -> [HW, B, C] (Transformer expects S, B, E)
        src_flatten = []
        mask_flatten = []
        pos_flatten = []
        spatial_shapes = []

        for l, (src, mask, pos_emb) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shapes.append((h, w))

            # Add level embedding
            src = src + self.level_embed[l].view(1, -1, 1, 1)

            src_flatten.append(src.flatten(2).permute(2, 0, 1))  # [HW, B, C]
            mask_flatten.append(mask.flatten(1))  # [B, HW]
            pos_flatten.append(pos_emb.flatten(2).permute(2, 0, 1))  # [HW, B, C]

        src_flatten = torch.cat(src_flatten, 0)
        mask_flatten = torch.cat(mask_flatten, 1)
        pos_flatten = torch.cat(pos_flatten, 0)

        # --- 2. Transformer Encoder ---
        # memory: [S, B, C]
        memory = self.transformer.encoder(
            src_flatten, src_key_padding_mask=mask_flatten, pos=pos_flatten
        )

        # --- 3. Auxiliary Heads (on Encoder Output) ---
        # We need to reshape memory back to spatial maps
        enc_outputs = {}
        if self.training:
            # Unflatten memory to feature maps
            memory_maps = []
            start_idx = 0
            memory_perm = memory.permute(1, 2, 0)  # [B, C, S]

            for h, w in spatial_shapes:
                end_idx = start_idx + h * w
                map_l = memory_perm[:, :, start_idx:end_idx].view(
                    -1, self.config.HIDDEN_DIM, h, w
                )
                memory_maps.append(map_l)
                start_idx = end_idx

            # ATSS Head
            atss_logits, atss_regs, atss_center = self.aux_atss(memory_maps)
            enc_outputs["atss"] = {
                "logits": atss_logits,
                "bbox_regs": atss_regs,
                "centerness": atss_center,
            }

            # RCNN Head
            # Generate pseudo-proposals from ATSS output (simplified: just use random or fixed grid for now
            # as full proposal generation + NMS is heavy).
            # Ideally, we decode ATSS boxes here. For this implementation, we skip explicit RCNN
            # forward in this block to save runtime/complexity unless strictly needed for loss.
            # We will just return the ATSS outputs which provide the dense supervision.

        # --- 4. Transformer Decoder ---
        bs = samples.tensors.shape[0]

        # Prepare Queries
        # Object Queries: [Num_Obj, 1, C] -> [Num_Obj, B, C]
        obj_queries = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        # Diagnosis Query: [1, 1, C] -> [1, B, C]
        diag_query = self.diagnosis_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        # Concatenate: [Num_Obj + 1, B, C]
        tgt = torch.cat([diag_query, obj_queries], dim=0)
        tgt_pos = torch.zeros_like(
            tgt
        )  # Learned queries usually don't need separate pos encoding passed here

        # Decoder
        hs = self.transformer.decoder(
            tgt,
            memory,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=mask_flatten,
            pos=pos_flatten,
            query_pos=tgt_pos,
        )
        # hs shape: [S, B, C] (Output of last layer)
        # We can also get intermediate layers if we used a custom loop, but nn.Transformer returns last layer by default
        # To support deep supervision, we would need the full stack. nn.Transformer usually returns only last.
        # We will proceed with the last layer.

        hs = hs.permute(1, 0, 2)  # [B, Num_Q, C]

        # --- 5. Prediction Heads ---
        # Split Diagnosis vs Object queries
        # Index 0 is Diagnosis, 1..N are Objects
        diag_out = hs[:, 0, :]  # [B, C]
        obj_out = hs[:, 1:, :]  # [B, Num_Obj, C]

        # Study Prediction
        pred_study = self.study_embed(diag_out)  # [B, 4]

        # Object Prediction
        pred_logits = self.class_embed(obj_out)  # [B, Num_Obj, Num_Classes + 1]
        pred_boxes = self.bbox_embed(
            obj_out
        ).sigmoid()  # [B, Num_Obj, 4] (normalized 0-1)

        out = {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "pred_study": pred_study,
            "enc_outputs": enc_outputs if self.training else None,
        }
        return out


def build_model(config=Config):
    model = CoDETR(config)
    return model

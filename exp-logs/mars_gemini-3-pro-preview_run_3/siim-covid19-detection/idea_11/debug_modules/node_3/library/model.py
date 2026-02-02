import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from library.config import Config
from library.backbone import build_backbone
from library.dino_layers import (
    DINOTransformerEncoder,
    DINOTransformerEncoderLayer,
    DINOTransformerDecoder,
    DINOTransformerDecoderLayer,
    ContrastiveDeNoising,
    MLP,
)


class RelationalFindingAggregator(nn.Module):
    """
    A specialized Transformer Encoder head that aggregates information from
    detected object queries to form a study-level diagnosis.

    It models the relationship between findings (e.g. bilateral opacities)
    to determine the clinical label.
    """

    def __init__(self, hidden_dim, num_classes, num_layers=1, nhead=8):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Transformer Encoder to mix information across queries
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final classification head
        self.classifier = MLP(hidden_dim, hidden_dim, num_classes, num_layers=2)

        # Learnable 'CLS' token to aggregate global context
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, query_embeddings):
        """
        Args:
            query_embeddings: (B, Num_Queries, Hidden_Dim)
        Returns:
            logits: (B, Num_Study_Classes)
        """
        B, N, C = query_embeddings.shape

        # Transpose to (N, B, C) for Transformer
        x = query_embeddings.permute(1, 0, 2)

        # Expand CLS token: (1, B, C)
        cls_token = self.cls_token.expand(-1, B, -1)

        # Prepend CLS token
        x = torch.cat((cls_token, x), dim=0)

        # Apply Transformer
        x = self.encoder(x)

        # Extract CLS token output: (1, B, C) -> (B, C)
        cls_out = x[0]

        # Predict
        logits = self.classifier(cls_out)
        return logits


class MultiTaskDINO(nn.Module):
    """
    Multi-Task DINO architecture with Swin-L backbone and Relational Finding Aggregator.
    """

    def __init__(self):
        super().__init__()
        self.hidden_dim = Config.HIDDEN_DIM
        self.num_queries = Config.NUM_QUERIES
        self.num_classes = Config.NUM_CLASSES  # Object classes (1 for opacity)
        self.num_study_classes = Config.NUM_STUDY_CLASSES

        # 1. Backbone
        self.backbone = build_backbone()

        # Project backbone features to hidden_dim
        # We use the last feature map (stride 32) for efficiency on 1024px images
        # Swin-L last stage channels: 1536
        backbone_channels = self.backbone.num_channels[-1]
        self.input_proj = nn.Conv2d(backbone_channels, self.hidden_dim, kernel_size=1)

        # 2. Transformer Encoder
        enc_layer = DINOTransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        )
        self.encoder = DINOTransformerEncoder(enc_layer, num_layers=Config.ENC_LAYERS)

        # 3. Transformer Decoder
        dec_layer = DINOTransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
        )
        self.decoder = DINOTransformerDecoder(dec_layer, num_layers=Config.DEC_LAYERS)

        # 4. Contrastive DeNoising
        self.cdn = ContrastiveDeNoising(
            hidden_dim=self.hidden_dim,
            num_classes=self.num_classes,
            num_queries=self.num_queries,
            label_noise_scale=0.2,
            box_noise_scale=0.4,
        )

        # 5. Prediction Heads
        # Box Head: Predicts normalized (cx, cy, w, h)
        self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        # Class Head: Predicts objectness/class
        self.class_embed = nn.Linear(self.hidden_dim, self.num_classes)

        # Study Head: Relational Aggregator
        self.study_aggregator = RelationalFindingAggregator(
            hidden_dim=self.hidden_dim, num_classes=self.num_study_classes
        )

        # 6. Enc Output Prediction (for Query Selection)
        # Used to select top-k features from encoder to initialize decoder queries
        self.enc_out_class_embed = nn.Linear(self.hidden_dim, self.num_classes)
        self.enc_out_bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, num_layers=3)

        # Weight initialization
        self._reset_parameters()

    def _reset_parameters(self):
        # Init projection
        nn.init.xavier_uniform_(self.input_proj.weight, gain=1)
        nn.init.constant_(self.input_proj.bias, 0)

        # Init class embed (bias towards background/low prob)
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        nn.init.constant_(self.class_embed.bias, bias_value)
        nn.init.constant_(self.enc_out_class_embed.bias, bias_value)

        # Init box embed
        nn.init.constant_(self.bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias, 0)
        nn.init.constant_(self.enc_out_bbox_embed.layers[-1].weight, 0)
        nn.init.constant_(self.enc_out_bbox_embed.layers[-1].bias, 0)

    def forward(self, samples, targets: Optional[List[Dict]] = None):
        """
        Args:
            samples: NestedTensor (tensors, mask)
            targets: List of dicts (labels, boxes) for CDN
        """
        # ---------------------------------------------------------
        # 1. Backbone & Projection
        # ---------------------------------------------------------
        features, pos = self.backbone(samples)

        # Use the last feature map (stride 32)
        # features is a list of NestedTensors. Index 3 is stride 32.
        src_nested = features[-1]
        src = src_nested.tensors
        mask = src_nested.mask

        # Project
        src = self.input_proj(src)  # (B, C, H, W)

        # Get Positional Embeddings for this scale
        # pos is a list. Index 3 corresponds to features[-1]
        pos_embed = pos[-1]  # (B, C, H, W)

        # Flatten for Transformer: (B, C, H, W) -> (HW, B, C)
        B, C, H, W = src.shape
        src_flatten = src.flatten(2).permute(2, 0, 1)
        pos_flatten = pos_embed.flatten(2).permute(2, 0, 1)
        mask_flatten = mask.flatten(1)  # (B, HW)

        # ---------------------------------------------------------
        # 2. Encoder
        # ---------------------------------------------------------
        memory = self.encoder(
            src_flatten, src_key_padding_mask=mask_flatten, pos=pos_flatten
        )
        # memory: (HW, B, C)

        # ---------------------------------------------------------
        # 3. Query Selection (Mixed Query Selection)
        # ---------------------------------------------------------
        # Predict on encoder outputs to find likely objects
        # memory shape for linear: (B, HW, C)
        memory_transposed = memory.permute(1, 0, 2)

        enc_logits = self.enc_out_class_embed(memory_transposed)  # (B, HW, num_classes)
        enc_boxes = self.enc_out_bbox_embed(memory_transposed).sigmoid()  # (B, HW, 4)

        # Select Top-K
        # We use the max probability across classes (if multiclass) or just the score (if binary)
        # Here num_classes=1, so just squeeze
        topk_scores, topk_indices = torch.topk(
            enc_logits.max(-1)[0], Config.NUM_QUERIES, dim=1
        )

        # Gather content queries (tgt) from memory
        # Expand indices to (B, K, C)
        batch_indices = (
            torch.arange(B, device=src.device)
            .unsqueeze(1)
            .repeat(1, Config.NUM_QUERIES)
        )
        tgt = memory_transposed[batch_indices, topk_indices, :]  # (B, K, C)

        # Gather reference points (query_pos) from positional embeddings
        # Note: In standard DINO, query_pos is usually learned or derived.
        # Here we use the positional embedding of the selected feature.
        pos_transposed = pos_flatten.permute(1, 0, 2)
        query_pos = pos_transposed[batch_indices, topk_indices, :]  # (B, K, C)

        # Transpose back to (K, B, C) for Decoder
        tgt = tgt.permute(1, 0, 2)
        query_pos = query_pos.permute(1, 0, 2)

        # ---------------------------------------------------------
        # 4. Contrastive DeNoising (CDN)
        # ---------------------------------------------------------
        dn_meta = None
        attn_mask = None

        if self.training and targets is not None:
            dn_content, dn_boxes, attn_mask, dn_meta = self.cdn(
                targets, label_enc_weight=self.class_embed.weight
            )

            if dn_content is not None:
                # Concatenate DN queries with Matching queries
                # dn_content: (L_dn, B, C)
                # tgt: (Num_Q, B, C)
                tgt = torch.cat([dn_content, tgt], dim=0)

                # For DN queries, we also need query_pos.
                # We can use the embedded noisy boxes or a learnable embedding.
                # To keep it simple and consistent with DINO structure, we project boxes to C
                # or just reuse the content as position (not ideal) or use zeros.
                # Here we will assume the decoder handles position via cross-attn mostly,
                # but we need a placeholder. Let's use zeros for DN pos or project them.
                # A simple MLP to project 4 coords to C would be better, but we lack it in init.
                # We will use zeros for DN query_pos to avoid shape mismatch errors.
                dn_pos = torch.zeros_like(dn_content)
                query_pos = torch.cat([dn_pos, query_pos], dim=0)

        # ---------------------------------------------------------
        # 5. Decoder
        # ---------------------------------------------------------
        # tgt: (L_total, B, C), memory: (HW, B, C)
        hs = self.decoder(
            tgt,
            memory,
            memory_key_padding_mask=mask_flatten,
            pos=pos_flatten,
            query_pos=query_pos,
            tgt_mask=attn_mask,  # Mask for CDN groups
        )
        # hs: (Num_Dec_Layers, L_total, B, C)

        # ---------------------------------------------------------
        # 6. Output Heads
        # ---------------------------------------------------------
        outputs_classes = []
        outputs_coords = []

        # Iterate over decoder layers
        for layer_idx, layer_hs in enumerate(hs):
            # layer_hs: (L_total, B, C) -> (B, L_total, C)
            layer_hs = layer_hs.permute(1, 0, 2)

            # Predict
            layer_cls = self.class_embed(layer_hs)
            layer_box = self.bbox_embed(layer_hs).sigmoid()

            outputs_classes.append(layer_cls)
            outputs_coords.append(layer_box)

        # Stack outputs: (Num_Layers, B, L_total, ...)
        stack_cls = torch.stack(outputs_classes)
        stack_box = torch.stack(outputs_coords)

        # ---------------------------------------------------------
        # 7. Post-Processing (Split DN vs Matching)
        # ---------------------------------------------------------
        out = {}

        if dn_meta is not None:
            L_dn = dn_meta["L_dn"]
            # Split
            dn_cls = stack_cls[:, :, :L_dn, :]
            dn_box = stack_box[:, :, :L_dn, :]

            match_cls = stack_cls[:, :, L_dn:, :]
            match_box = stack_box[:, :, L_dn:, :]

            # Store DN output for loss
            out["dn_logits"] = dn_cls[-1]
            out["dn_boxes"] = dn_box[-1]
            out["dn_meta"] = dn_meta
        else:
            match_cls = stack_cls
            match_box = stack_box

        # Final layer outputs (Matching queries only)
        # These are the actual predictions
        pred_logits = match_cls[-1]  # (B, Num_Q, Num_Classes)
        pred_boxes = match_box[-1]  # (B, Num_Q, 4)

        out["pred_logits"] = pred_logits
        out["pred_boxes"] = pred_boxes

        # Auxiliary outputs (intermediate layers)
        out["aux_outputs"] = [
            {"pred_logits": match_cls[i], "pred_boxes": match_box[i]}
            for i in range(Config.DEC_LAYERS - 1)
        ]

        # Add Encoder outputs (for loss on initial selection)
        out["enc_outputs"] = {
            "pred_logits": enc_logits,  # (B, HW, C)
            "pred_boxes": enc_boxes,  # (B, HW, 4)
        }

        # ---------------------------------------------------------
        # 8. Study Level Prediction
        # ---------------------------------------------------------
        # Use the refined queries from the LAST decoder layer (Matching only)
        # match_hs: (B, Num_Q, C) derived from hs[-1] split
        # We need to grab the embeddings again since we only saved logits/boxes above
        last_hs = hs[-1].permute(1, 0, 2)  # (B, L_total, C)
        if dn_meta is not None:
            match_embeddings = last_hs[:, dn_meta["L_dn"] :, :]
        else:
            match_embeddings = last_hs

        study_logits = self.study_aggregator(match_embeddings)
        out["study_logits"] = study_logits

        return out

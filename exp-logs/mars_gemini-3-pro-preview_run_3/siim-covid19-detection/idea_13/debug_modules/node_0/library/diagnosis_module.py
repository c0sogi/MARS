import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from typing import List, Dict, Optional, Tuple

from library.config import Config
from library.utils import box_cxcywh_to_xyxy


class RoITransformerDiagnosis(nn.Module):
    """
    RoI-Transformer Diagnosis Module for Study-Level Classification.

    Integrates global context from the backbone with local features extracted
    from DINO-predicted bounding boxes to perform relational reasoning for
    diagnosis (Negative, Typical, Indeterminate, Atypical).
    """

    def __init__(
        self,
        config: type = Config,
        in_channels_list: List[int] = [
            384,
            768,
            1536,
        ],  # Default for Swin-L indices 1, 2, 3
        top_k: int = 50,
    ):
        super().__init__()
        self.img_size = config.IMG_SIZE
        self.hidden_dim = config.ROI_HEAD_DIM
        self.top_k = top_k
        self.roi_size = config.ROI_ALIGN_SIZE  # (7, 7)
        self.num_classes = config.NUM_CLASSES_STUDY

        # 1. Feature Projections
        # Projects backbone features to hidden_dim for RoIAlign and Global Context
        # We assume input features correspond to the order in in_channels_list
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_c, self.hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, self.hidden_dim),
                )
                for in_c in in_channels_list
            ]
        )

        # 2. RoI Align
        # Extracts features from projected maps.
        # We use keys '0', '1', '2' to map to the projected features.
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2"], output_size=self.roi_size, sampling_ratio=2
        )

        # 3. RoI Feature Encoder
        # Flattens the 7x7 spatial features into a vector
        roi_feat_len = self.hidden_dim * self.roi_size[0] * self.roi_size[1]
        self.roi_projection = nn.Linear(roi_feat_len, self.hidden_dim)

        # 4. Global Context Encoder
        # Projects the Global Average Pooled feature from the last scale
        self.global_projection = nn.Linear(self.hidden_dim, self.hidden_dim)

        # 5. Box Coordinate Embedding
        # Embeds (cx, cy, w, h) to provide spatial information to the transformer
        self.box_embedding = nn.Linear(4, self.hidden_dim)

        # 6. Transformer Encoder
        # Performs relational reasoning between the global context and local findings
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=config.ROI_NUM_HEADS,
            dim_feedforward=self.hidden_dim * 4,
            dropout=config.ROI_DROPOUT,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=2  # Lightweight encoder
        )

        # 7. Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.ROI_DROPOUT),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        # Token Type Embeddings: 0 for Global, 1 for RoI
        self.token_type_embed = nn.Embedding(2, self.hidden_dim)

        # Init weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        features: List[torch.Tensor],
        pred_boxes: torch.Tensor,
        pred_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            features: List of backbone features (B, C, H, W).
            pred_boxes: DINO predicted boxes (B, N, 4) in (cx, cy, w, h) normalized [0, 1].
            pred_logits: DINO predicted logits (B, N, 1) or (B, N, num_classes).

        Returns:
            study_logits: (B, 4) unnormalized logits for the study labels.
        """
        B = features[0].shape[0]
        device = features[0].device

        # 1. Project Features
        # Map names '0', '1', '2' correspond to the order in features list
        proj_feats = []
        features_dict = {}
        for i, (feat, proj) in enumerate(zip(features, self.projections)):
            p = proj(feat)
            proj_feats.append(p)
            features_dict[str(i)] = p

        # 2. Select Top-K Boxes
        # Apply sigmoid to get confidence scores
        # pred_logits shape is (B, N, 1) for binary detection
        scores = pred_logits.sigmoid().squeeze(-1)  # (B, N)

        # Determine K
        k = min(self.top_k, scores.shape[1])

        # Get indices of top-k boxes
        topk_scores, topk_indices = torch.topk(scores, k, dim=1)

        # Gather boxes: (B, K, 4)
        # topk_indices: (B, K) -> expand to (B, K, 4)
        idx_expanded = topk_indices.unsqueeze(-1).expand(-1, -1, 4)
        selected_boxes = torch.gather(pred_boxes, 1, idx_expanded)

        # 3. Prepare Boxes for RoIAlign
        # RoIAlign expects absolute coordinates (x1, y1, x2, y2)
        # and a list of tensors [K, 4] per image.
        rois_list = []

        # Scale factor to convert normalized [0,1] to absolute pixels [0, IMG_SIZE]
        scale_tensor = torch.tensor(
            [self.img_size, self.img_size, self.img_size, self.img_size], device=device
        )

        for b in range(B):
            # (K, 4)
            box_norm_cxcywh = selected_boxes[b]
            box_norm_xyxy = box_cxcywh_to_xyxy(box_norm_cxcywh)

            # Clamp to [0, 1] to be safe
            box_norm_xyxy = box_norm_xyxy.clamp(0.0, 1.0)

            # Scale to absolute pixels
            box_abs = box_norm_xyxy * scale_tensor
            rois_list.append(box_abs)

        # 4. Extract RoI Features
        # We must provide image_shapes so MultiScaleRoIAlign can calculate scales
        # relative to the original image size (which we scaled boxes to).
        image_shapes = [(self.img_size, self.img_size)] * B

        # output: (B*K, C, 7, 7)
        roi_features = self.roi_align(features_dict, rois_list, image_shapes)

        # 5. Process RoI Features
        # Flatten and project: (B*K, C*49) -> (B*K, dim)
        roi_features = roi_features.flatten(1)
        roi_tokens = self.roi_projection(roi_features)
        roi_tokens = roi_tokens.view(B, k, self.hidden_dim)

        # Add Box Spatial Embeddings
        # selected_boxes is (B, K, 4) normalized
        box_emb = self.box_embedding(selected_boxes)  # (B, K, dim)
        roi_tokens = roi_tokens + box_emb

        # Add Token Type Embedding (Type 1 for RoI)
        roi_tokens = roi_tokens + self.token_type_embed(torch.tensor(1, device=device))

        # 6. Global Context Token
        # Use the last feature map (stride 32, index '2')
        # (B, C, H, W) -> GAP -> (B, C, 1, 1) -> (B, C)
        global_feat = proj_feats[-1]
        global_token = F.adaptive_avg_pool2d(global_feat, (1, 1)).flatten(1)
        global_token = self.global_projection(global_token)  # (B, dim)

        # Add Token Type Embedding (Type 0 for Global)
        global_token = global_token + self.token_type_embed(
            torch.tensor(0, device=device)
        )

        # 7. Concatenate Sequence
        # Sequence: [Global, RoI_1, ... RoI_K]
        # (B, 1, dim) concat (B, K, dim) -> (B, K+1, dim)
        global_token = global_token.unsqueeze(1)
        sequence = torch.cat([global_token, roi_tokens], dim=1)

        # 8. Transformer Reasoning
        output_sequence = self.transformer(sequence)

        # 9. Classification
        # Use the first token (corresponding to Global Context) for classification
        cls_token = output_sequence[:, 0, :]
        logits = self.classifier(cls_token)

        return logits

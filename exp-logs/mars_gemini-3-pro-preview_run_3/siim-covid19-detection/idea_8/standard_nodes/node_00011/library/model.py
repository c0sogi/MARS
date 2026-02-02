import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DeformableDetrConfig, DeformableDetrModel
from library.config import Config


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (Feed-Forward Network) used for prediction heads.
    """

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


class MultiTaskDeformableDETR(nn.Module):
    """
    Multi-Task Deformable DETR with Unified Query Interaction.

    This model extends the standard Deformable DETR architecture by appending a
    specific 'Study Query' to the set of object queries. This allows the model
    to simultaneously perform object detection (opacity localization) and
    study-level classification (diagnosis) within a single transformer decoder,
    enabling the study prediction to attend to detected objects.
    """

    def __init__(self):
        super().__init__()

        # 1. Configuration
        # Initialize config based on hyperparameters in library.config
        self.config = DeformableDetrConfig(
            d_model=Config.HIDDEN_DIM,
            encoder_layers=Config.NUM_ENCODER_LAYERS,
            decoder_layers=Config.NUM_DECODER_LAYERS,
            encoder_attention_heads=Config.NHEADS,
            decoder_attention_heads=Config.NHEADS,
            encoder_ffn_dim=Config.DIM_FEEDFORWARD,
            decoder_ffn_dim=Config.DIM_FEEDFORWARD,
            dropout=Config.DROPOUT,
            num_feature_levels=Config.NUM_FEATURE_LEVELS,
            encoder_n_points=Config.ENC_N_POINTS,
            decoder_n_points=Config.DEC_N_POINTS,
            two_stage=False,  # Single-stage for unified query simplicity
            auxiliary_loss=False,
            num_queries=Config.NUM_OBJECT_QUERIES + Config.NUM_STUDY_QUERIES,
            backbone=Config.BACKBONE,
            use_timm_backbone=True,
        )

        # 2. Core Transformer Model
        # Initialized from scratch (random weights) as per environment constraints
        self.model = DeformableDetrModel(self.config)

        # 3. Query Embeddings
        # We define learnable embeddings for the unified query set.
        # Shape: (Total_Queries, Hidden_Dim * 2)
        # The *2 width allows the model to split into content and position embeddings
        # internally, or use them as learnable positional anchors.
        self.num_queries = Config.NUM_OBJECT_QUERIES + Config.NUM_STUDY_QUERIES
        self.query_embeds = nn.Embedding(self.num_queries, Config.HIDDEN_DIM * 2)

        # 4. Prediction Heads

        # A. Object Detection Heads (Applied to the first N queries)
        # Class Head: Predicts 'opacity' (1) or 'background' (0) -> 2 outputs effectively
        # We output num_classes + 1 to account for the 'no object' class.
        self.class_embed = nn.Linear(Config.HIDDEN_DIM, Config.NUM_OBJECT_CLASSES + 1)

        # Box Head: Predicts normalized (cx, cy, w, h)
        self.bbox_embed = MLP(Config.HIDDEN_DIM, Config.HIDDEN_DIM, 4, 3)

        # B. Study Classification Head (Applied to the last query)
        # Predicts one of the study labels (Negative, Typical, Indeterminate, Atypical)
        self.study_embed = MLP(
            Config.HIDDEN_DIM, Config.HIDDEN_DIM, Config.NUM_STUDY_CLASSES, 3
        )

        # 5. Weight Initialization
        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights, specifically setting priors for classification layers
        to improve training stability with Focal Loss.
        """
        # Xavier initialization for linear layers
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Bias initialization for object classification
        # Set bias to -log((1-p)/p) for low prior probability of foreground
        prior_prob = 0.01
        bias_value = -torch.log(torch.tensor((1 - prior_prob) / prior_prob))
        self.class_embed.bias.data = (
            torch.ones(Config.NUM_OBJECT_CLASSES + 1) * bias_value
        )

        # Initialize box regression head to output near-zero corrections initially
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, pixel_values, pixel_mask=None):
        """
        Args:
            pixel_values (Tensor): Images of shape (Batch, 3, H, W)
            pixel_mask (Tensor, optional): Mask of shape (Batch, H, W),
                                           where 1 indicates padding.
        Returns:
            dict: Contains 'pred_logits', 'pred_boxes', 'pred_study_logits'
        """
        batch_size = pixel_values.shape[0]

        # 1. Prepare Query Embeddings
        # Expand the learnable embeddings to match the batch size
        # Shape: (Batch, Total_Queries, Hidden_Dim * 2)
        query_embeds = self.query_embeds.weight.unsqueeze(0).repeat(batch_size, 1, 1)

        # 2. Transformer Forward Pass
        outputs = self.model(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            decoder_inputs_embeds=query_embeds,
        )

        # 3. Extract Hidden States
        # last_hidden_state shape: (Batch, Total_Queries, Hidden_Dim)
        hs = outputs.last_hidden_state

        # 4. Split Streams (Object vs Study)
        # The first NUM_OBJECT_QUERIES are responsible for object detection
        object_hs = hs[:, : Config.NUM_OBJECT_QUERIES, :]

        # The last query is responsible for study classification
        study_hs = hs[:, Config.NUM_OBJECT_QUERIES :, :]

        # 5. Prediction Heads

        # Object Predictions
        outputs_class = self.class_embed(object_hs)
        outputs_coord = self.bbox_embed(
            object_hs
        ).sigmoid()  # Coordinates must be [0, 1]

        # Study Predictions
        # Squeeze the singleton dimension: (Batch, 1, Hidden) -> (Batch, Hidden)
        study_hs_squeezed = study_hs.squeeze(1)
        outputs_study = self.study_embed(study_hs_squeezed)

        return {
            "pred_logits": outputs_class,  # Shape: (B, N_Obj, Num_Classes + 1)
            "pred_boxes": outputs_coord,  # Shape: (B, N_Obj, 4)
            "pred_study_logits": outputs_study,  # Shape: (B, Num_Study_Classes)
        }

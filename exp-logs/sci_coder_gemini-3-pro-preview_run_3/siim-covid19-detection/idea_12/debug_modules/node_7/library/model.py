import torch
import torch.nn as nn
from collections import namedtuple
from transformers import AutoModelForObjectDetection, AutoConfig
from library.config import Config
from library.reasoning_head import DualStreamReasoningModule


class SwinBackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, pixel_values, pixel_mask=None):
        # Backbone returns (features, position_embeddings)
        features, position_embeddings = self.backbone(pixel_values, pixel_mask)

        new_features = []
        BackboneElement = namedtuple("BackboneElement", ["pixel_values", "pixel_mask"])

        for feat in features:
            # Extract tensor and mask from BackboneElement
            p_values = feat.pixel_values
            p_mask = feat.pixel_mask

            # Swin outputs (N, H, W, C), convert to (N, C, H, W)
            if p_values.ndim == 4:
                p_values = p_values.permute(0, 3, 1, 2)

            new_features.append(BackboneElement(p_values, p_mask))

        return new_features, position_embeddings


class MultiTaskDINO(nn.Module):
    """
    Multi-Task DINO with Dual-Stream Geometric-Semantic Reasoning.

    Wraps a pre-trained DINO (with Swin-L backbone) for object detection and
    adds a custom reasoning head for study-level classification.
    """

    def __init__(self):
        super(MultiTaskDINO, self).__init__()

        # 1. Load and Configure DINO
        # We use the Swin-Large variant of DINO as the base
        checkpoint = "IDEA-Research/dino-swin-large"

        try:
            self.config = AutoConfig.from_pretrained(checkpoint)
            # Update Config for specific task requirements
            self.config.num_labels = Config.NUM_DETECTION_CLASSES
            self.config.num_queries = Config.NUM_QUERIES
            self.config.use_timm_backbone = True
            self.config.backbone = Config.BACKBONE_NAME
        except Exception:
            # Fallback to a default config if checkpoint fetch fails (e.g. no internet)
            print(
                f"Warning: Could not load config from {checkpoint}. Creating default config."
            )
            try:
                from transformers import DinoConfig

                self.config = DinoConfig(
                    use_timm_backbone=True,
                    backbone=Config.BACKBONE_NAME,
                    backbone_kwargs={
                        "out_indices": [0, 1, 2, 3],
                        "img_size": Config.IMG_SIZE,
                    },
                    num_labels=Config.NUM_DETECTION_CLASSES,
                    num_queries=Config.NUM_QUERIES,
                    d_model=Config.HIDDEN_DIM,
                    encoder_layers=Config.ENC_LAYERS,
                    decoder_layers=Config.DEC_LAYERS,
                    encoder_attention_heads=Config.NHEADS,
                    decoder_attention_heads=Config.NHEADS,
                    encoder_ffn_dim=Config.DIM_FEEDFORWARD,
                    decoder_ffn_dim=Config.DIM_FEEDFORWARD,
                    dropout=Config.DROPOUT,
                )
            except ImportError:
                # Fallback for older transformers versions
                from transformers import DeformableDetrConfig

                self.config = DeformableDetrConfig(
                    use_timm_backbone=True,
                    backbone=Config.BACKBONE_NAME,
                    backbone_kwargs={
                        "out_indices": [0, 1, 2, 3],
                        "img_size": Config.IMG_SIZE,
                    },
                    num_labels=Config.NUM_DETECTION_CLASSES,
                    num_queries=Config.NUM_QUERIES,
                    d_model=Config.HIDDEN_DIM,
                    encoder_layers=Config.ENC_LAYERS,
                    decoder_layers=Config.DEC_LAYERS,
                    encoder_attention_heads=Config.NHEADS,
                    decoder_attention_heads=Config.NHEADS,
                    encoder_ffn_dim=Config.DIM_FEEDFORWARD,
                    decoder_ffn_dim=Config.DIM_FEEDFORWARD,
                    dropout=Config.DROPOUT,
                )

        # Load the model
        # ignore_mismatched_sizes is crucial because we are changing num_labels and num_queries
        try:
            self.dino = AutoModelForObjectDetection.from_pretrained(
                checkpoint, config=self.config, ignore_mismatched_sizes=True
            )
        except Exception as e:
            print(
                f"Error loading pretrained model: {e}. Initializing with random weights."
            )
            if self.config is not None:
                self.dino = AutoModelForObjectDetection.from_config(self.config)
            else:
                raise RuntimeError("Could not load model configuration or weights.")

        # 2. Initialize Reasoning Head
        self.reasoning_head = DualStreamReasoningModule()

        # 3. Feature Projection (if necessary)
        # Ensure DINO hidden dim matches Reasoning Head hidden dim
        self.dino_dim = self.config.d_model
        self.reasoning_dim = Config.HIDDEN_DIM

        if self.dino_dim != self.reasoning_dim:
            self.project_queries = nn.Linear(self.dino_dim, self.reasoning_dim)
        else:
            self.project_queries = nn.Identity()

        # FIX: Wrap backbone to handle Swin Transformer's channel-last output
        if (
            "swin" in Config.BACKBONE_NAME
            and hasattr(self.dino, "model")
            and hasattr(self.dino.model, "backbone")
        ):
            self.dino.model.backbone = SwinBackboneWrapper(self.dino.model.backbone)

    def forward(self, pixel_values, pixel_mask=None, labels=None):
        """
        Forward pass.

        Args:
            pixel_values (Tensor): Images [Batch, 3, H, W]
            pixel_mask (Tensor, optional): Mask for padding [Batch, H, W]
            labels (List, optional): Targets (unused in forward, handled by loss criterion externally)

        Returns:
            dict: {
                'pred_logits': [Batch, Num_Queries, Num_Classes],
                'pred_boxes': [Batch, Num_Queries, 4],
                'study_logits': [Batch, Num_Study_Classes]
            }
        """
        # Pass through DINO
        # We need hidden states to access the updated query embeddings for the reasoning head
        outputs = self.dino(
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        # Extract Detection Outputs
        # pred_logits: [Batch, Num_Queries, Num_Classes]
        # pred_boxes: [Batch, Num_Queries, 4] (sigmoid normalized coordinates)
        pred_logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        # Extract Query Embeddings
        # decoder_hidden_states is a tuple of states for each layer.
        # We take the last one: [Batch, Num_Queries, Hidden_Dim]
        last_hidden_state = outputs.decoder_hidden_states[-1]

        # Project if dimensions mismatch
        query_embeds = self.project_queries(last_hidden_state)

        # Pass through Dual-Stream Reasoning Head
        # Inputs:
        #   query_embeds: Semantic features of potential objects
        #   pred_boxes: Geometric configuration of potential objects
        study_logits = self.reasoning_head(query_embeds, pred_boxes)

        return {
            "pred_logits": pred_logits,
            "pred_boxes": pred_boxes,
            "study_logits": study_logits,
        }

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import List, Dict, Tuple, Optional

from library.config import Config


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position embedding for Transformers.
    Adapted from standard DETR/DINO implementations.
    """

    def __init__(
        self, num_pos_feats: int = 64, temperature: int = 10000, normalize: bool = False
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W).
            mask: Boolean mask of shape (B, H, W). True indicates padding (ignored).
                  If None, assumes no padding (all pixels valid).

        Returns:
            pos: Position embeddings of shape (B, C, H, W).
                 Note: Output C will be num_pos_feats * 2 (y and x coords).
        """
        if mask is None:
            mask = torch.zeros(
                (x.shape[0], x.shape[2], x.shape[3]), dtype=torch.bool, device=x.device
            )

        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class Backbone(nn.Module):
    """
    Backbone wrapper for timm models (specifically Swin Transformer).
    Extracts multi-scale feature maps.
    """

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3),
    ):
        """
        Args:
            backbone_name: Name of the timm model (e.g., 'swin_large_patch4_window12_384').
            pretrained: Whether to load pretrained weights.
            out_indices: Indices of the stages to return features from.
                         For Swin: 0=stride 4, 1=stride 8, 2=stride 16, 3=stride 32.
        """
        super().__init__()
        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )
        # Capture the channel counts for the selected indices
        self.num_channels = self.body.feature_info.channels()

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            xs: List of feature maps from selected stages.
        """
        xs = self.body(x)
        return xs


class Joiner(nn.Module):
    """
    Combines Backbone and Position Embedding.
    Returns features, masks, and position embeddings required by DINO/DETR.
    """

    def __init__(self, backbone: Backbone, position_embedding: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
        """
        Args:
            x: Input tensor (B, 3, H, W)

        Returns:
            out: List of feature maps (B, C, H_feat, W_feat)
            masks: List of masks (B, H_feat, W_feat) - True indicates padding
            pos: List of position embeddings (B, C_pos, H_feat, W_feat)
        """
        # Get multi-scale features from backbone
        xs = self.backbone(x)

        out: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        pos: List[torch.Tensor] = []

        for x_feat in xs:
            out.append(x_feat)

            # Generate mask: assumes input x is fully valid (no padding mask passed in)
            # In this pipeline, images are resized/padded via Albumentations, so the tensor is dense.
            # We treat all pixels as valid for the attention mask (all False).
            mask = torch.zeros(
                (x_feat.shape[0], x_feat.shape[2], x_feat.shape[3]),
                dtype=torch.bool,
                device=x_feat.device,
            )
            masks.append(mask)

            # Generate position encoding
            pos.append(self.position_embedding(x_feat, mask))

        return out, masks, pos


def build_backbone(config: Optional[object] = None, hidden_dim: int = 256) -> Joiner:
    """
    Factory function to build the backbone with position embedding.

    Args:
        config: Configuration object containing BACKBONE name.
                If None, uses the global Config class.
        hidden_dim: Dimension for position embeddings (usually matches transformer hidden dim).
                    num_pos_feats will be hidden_dim // 2.

    Returns:
        model: Joiner module containing Backbone and PositionEmbeddingSine.
    """
    if config is None:
        config = Config

    # Instantiate Position Embedding
    # num_pos_feats is half of hidden_dim because we concatenate x and y embeddings
    position_embedding = PositionEmbeddingSine(
        num_pos_feats=hidden_dim // 2, normalize=True
    )

    # Instantiate Backbone
    # We request indices (1, 2, 3) for Swin to get strides 8, 16, 32
    # Swin stages: 0 (stride 4), 1 (stride 8), 2 (stride 16), 3 (stride 32)
    backbone = Backbone(
        backbone_name=config.BACKBONE,
        pretrained=True,
        out_indices=(1, 2, 3),
    )

    # Combine
    model = Joiner(backbone, position_embedding)

    return model

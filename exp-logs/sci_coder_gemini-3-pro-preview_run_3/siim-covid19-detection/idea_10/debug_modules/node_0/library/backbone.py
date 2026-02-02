import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math
from typing import List, Dict, Optional

from library.config import Config


class NestedTensor:
    """
    A simple container for a tensor and a corresponding mask.
    Used to bundle the feature map with its padding mask.
    """

    def __init__(self, tensors: torch.Tensor, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is All You Need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
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


class SwinBackbone(nn.Module):
    """
    Swin Transformer Backbone using timm.
    Extracts multi-scale features for Deformable DETR.
    """

    def __init__(self, backbone_name, pretrained=True):
        super().__init__()

        # Create Swin Transformer model
        # out_indices=(1, 2, 3) corresponds to strides 8, 16, 32
        # These are the standard feature maps used in Deformable DETR (C3, C4, C5)
        self.model = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(1, 2, 3),
        )

        # Extract channel information for the projection layers in the transformer
        # feature_info is a list of dicts with 'num_chs', 'reduction', 'module'
        self.num_channels = [info["num_chs"] for info in self.model.feature_info]
        self.strides = [info["reduction"] for info in self.model.feature_info]

        print(f"Initialized {backbone_name}")
        print(f"Feature Channels: {self.num_channels}")
        print(f"Feature Strides: {self.strides}")

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor containing image tensor (B, C, H, W) and mask
        Returns:
            xs: Dict of feature maps
        """
        x = tensor_list.tensors

        # timm returns a list of tensors
        outs = self.model(x)

        # Return as a dictionary to match DETR expectation often seen in implementations
        # mapping layer name/index to NestedTensor
        ret = {}
        for i, out in enumerate(outs):
            m = tensor_list.mask
            assert m is not None

            # Resize mask to match feature map size
            # Mask is (B, H, W), feature is (B, C, h, w)
            # We use nearest neighbor interpolation for the boolean mask
            mask = F.interpolate(m[None].float(), size=out.shape[-2:]).to(torch.bool)[0]

            ret[f"{i}"] = NestedTensor(out, mask)

        return ret


class Joiner(nn.Module):
    """
    Combines the backbone and the position embedding.
    """

    def __init__(self, backbone, position_embedding):
        super().__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding
        self.num_channels = backbone.num_channels
        self.strides = backbone.strides

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor
        Returns:
            features: List of NestedTensors (feature maps)
            pos: List of Tensors (position embeddings)
        """
        xs = self.backbone(tensor_list)

        out: List[NestedTensor] = []
        pos: List[torch.Tensor] = []

        # Iterate over the dictionary returned by backbone (keys are "0", "1", "2")
        for name, x in sorted(xs.items()):
            out.append(x)
            # position encoding
            pos.append(self.position_embedding(x))

        return out, pos


def build_backbone(config=None):
    """
    Factory function to build the backbone with position embeddings.
    """
    if config is None:
        # Fallback to default Config class if not provided
        config = Config

    # 1. Build Position Embedding
    # Hidden dim is usually 256 for Deformable DETR.
    # Position embedding size is hidden_dim / 2 (since it concatenates x and y).
    N_steps = config.HIDDEN_DIM // 2
    position_embedding = PositionEmbeddingSine(num_pos_feats=N_steps, normalize=True)

    # 2. Build Backbone
    backbone = SwinBackbone(
        backbone_name=config.BACKBONE_NAME, pretrained=config.BACKBONE_PRETRAINED
    )

    # 3. Combine
    model = Joiner(backbone, position_embedding)

    return model

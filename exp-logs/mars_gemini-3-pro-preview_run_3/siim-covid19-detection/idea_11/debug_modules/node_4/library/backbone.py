import torch
import torch.nn.functional as F
from torch import nn
import math
import timm
from typing import List, Dict, Optional
from library.config import Config


class NestedTensor:
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
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


class Backbone(nn.Module):
    """
    Backbone class that wraps a timm model.
    """

    def __init__(
        self, name: str, pretrained: bool = True, out_indices: tuple = (1, 2, 3)
    ):
        super().__init__()
        # Create the timm model with features_only=True to get intermediate layers
        # out_indices selects which stages to output.
        # Swin usually has 4 stages. indices (0, 1, 2, 3) correspond to strides (4, 8, 16, 32).
        self.body = timm.create_model(
            name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            img_size=Config.IMG_SIZE,
        )

        # Get channels for the selected feature maps
        self.num_channels = self.body.feature_info.channels()
        self.out_indices = out_indices

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor containing image tensor (B, C, H, W) and mask (B, H, W)
        Returns:
            xs: Dict of NestedTensors for each feature level
        """
        # Forward pass through the backbone
        # timm returns a list of tensors
        xs = self.body(tensor_list.tensors)

        out = {}
        for i, x in enumerate(xs):
            # Swin Transformer in timm returns NHWC, convert to NCHW if needed
            if x.shape[-1] == self.num_channels[i]:
                x = x.permute(0, 3, 1, 2).contiguous()

            # Resize mask to match feature map size
            m = tensor_list.mask
            assert m is not None
            # Interpolate mask to feature map size (nearest neighbor)
            # mask is (B, H, W), need (B, 1, H, W) for interpolate
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]

            out[f"{i}"] = NestedTensor(x, mask)

        return out


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        """
        Args:
            tensor_list: NestedTensor
        Returns:
            features: List of NestedTensors
            pos: List of positional embeddings
        """
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(args=None):
    """
    Builds the backbone with positional embeddings.
    """
    # Use Config for settings
    backbone_name = Config.BACKBONE_NAME

    # We use the hidden dimension from config to determine pos embedding size
    # DINO/DETR usually splits hidden_dim into 2 for x and y sine embeddings
    hidden_dim = Config.HIDDEN_DIM

    position_embedding = PositionEmbeddingSine(
        num_pos_feats=hidden_dim // 2, normalize=True
    )

    # Instantiate Backbone
    # Using indices (1, 2, 3) for multi-scale (strides 8, 16, 32)
    # Or (0, 1, 2, 3) if we want stride 4 as well.
    # Typically DINO uses 4 scales. Swin-L has 4 stages.
    backbone = Backbone(name=backbone_name, pretrained=True, out_indices=(0, 1, 2, 3))

    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels

    return model

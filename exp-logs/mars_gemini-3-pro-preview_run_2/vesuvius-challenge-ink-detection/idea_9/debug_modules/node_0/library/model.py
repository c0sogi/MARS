import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from library.config import CFG


class WideContextSegFormer(nn.Module):
    """
    Wide-Context SegFormer (MiT-B4) with Learnable Z-Compression.

    This model ingests a 5-channel input representing overlapping Z-slabs of the scroll fragment.
    It uses a learnable 1x1 convolution adapter to compress these 5 channels into the 3 channels
    expected by the pretrained SegFormer backbone.

    The adapter is initialized to an identity mapping for the central 3 channels, ensuring that
    at epoch 0, the model behaves like a standard 3-channel model, gradually learning to utilize
    the outer Z-context.
    """

    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

        # Load the pretrained SegFormer model
        # We use ignore_mismatched_sizes=True because we are loading an encoder-only checkpoint
        # (nvidia/mit-b4) or a checkpoint with different classes into a model with num_labels=1.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            cfg.model_name, num_labels=cfg.num_classes, ignore_mismatched_sizes=True
        )

        # Learnable Channel Compressor (Adapter)
        # Projects input_channels (5) -> 3 (RGB for backbone)
        self.adapter = nn.Conv2d(
            in_channels=cfg.input_channels, out_channels=3, kernel_size=1, bias=True
        )

        # Initialize the adapter with specific identity mapping logic
        self._init_adapter()

    def _init_adapter(self):
        """
        Initializes the adapter layer.

        Strategy:
        - Weights for Input Channels 2, 3, 4 (indices 1, 2, 3) map to Output Channels 1, 2, 3 (indices 0, 1, 2).
        - Weights for Input Channels 1, 5 (indices 0, 4) are initialized to 0.
        - Bias is initialized to 0.

        This ensures the model starts by focusing on the central slabs, preserving the stability
        of a 3-channel baseline.
        """
        with torch.no_grad():
            # Reset all weights and biases to 0
            self.adapter.weight.zero_()
            self.adapter.bias.zero_()

            # Set Identity Mapping for central channels
            # Adapter Weight Shape: (out_channels, in_channels, k_h, k_w) -> (3, 5, 1, 1)

            # Input Index 1 (2nd slab) -> Output Index 0 (R)
            self.adapter.weight[0, 1, 0, 0] = 1.0

            # Input Index 2 (3rd slab, Center) -> Output Index 1 (G)
            self.adapter.weight[1, 2, 0, 0] = 1.0

            # Input Index 3 (4th slab) -> Output Index 2 (B)
            self.adapter.weight[2, 3, 0, 0] = 1.0

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 5, Height, Width)

        Returns:
            torch.Tensor: Logits of shape (Batch, 1, Height, Width)
        """
        # 1. Adapt 5-channel input to 3-channel backbone input
        # x: (B, 5, H, W) -> (B, 3, H, W)
        x = self.adapter(x)

        # 2. Pass through SegFormer
        # SegFormer expects 'pixel_values'
        outputs = self.model(pixel_values=x)

        # 3. Extract logits
        # Output shape from backbone is usually (B, num_labels, H/4, W/4)
        logits = outputs.logits

        # 4. Upsample to original image size
        # We use bilinear interpolation to match the input resolution
        upsampled_logits = F.interpolate(
            logits,
            size=(self.cfg.image_size, self.cfg.image_size),
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class UnifiedSegFormer(nn.Module):
    """
    Unified Translation-Invariant SegFormer model.
    Wraps the Hugging Face SegformerForSemanticSegmentation class.

    Architecture:
        - Backbone: MiT-B2 (nvidia/mit-b2)
        - Decoder: Standard All-MLP Decoder
        - Input: 3-channel (RGB interface)
        - Output: Binary segmentation logits upsampled to input resolution
    """

    def __init__(self):
        super(UnifiedSegFormer, self).__init__()

        # Load the pretrained SegFormer model
        # We set num_labels=1 for binary segmentation (Ink vs No-Ink).
        # ignore_mismatched_sizes=True is required because the pretrained decoder head
        # (usually 150 classes for ADE20k) is replaced with a 1-class head.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.BACKBONE,
            num_labels=Config.NUM_CLASSES,
            num_channels=Config.IN_CHANNELS,
            ignore_mismatched_sizes=True,
        )

    def forward(self, images):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input images of shape (B, 3, H, W).

        Returns:
            torch.Tensor: Predicted logits of shape (B, 1, H, W).
        """
        # Capture original input dimensions for upsampling
        input_h, input_w = images.shape[-2:]

        # Pass through the SegFormer model
        # The model expects the argument 'pixel_values'
        outputs = self.model(pixel_values=images)

        # Extract logits from the output object
        # The raw logits from SegFormer are typically 1/4th of the input resolution
        logits = outputs.logits

        # Upsample logits to match the input resolution using bilinear interpolation
        upsampled_logits = F.interpolate(
            logits, size=(input_h, input_w), mode="bilinear", align_corners=False
        )

        return upsampled_logits

import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class SegFormerB3(nn.Module):
    """
    SegFormer model with MiT-B3 backbone for binary segmentation.
    Wraps the Hugging Face Transformers implementation.
    """

    def __init__(self):
        super().__init__()

        # Load the pretrained model with the specified backbone.
        # We set num_labels to 1 for binary ink detection.
        # ignore_mismatched_sizes=True is required because we are replacing the
        # original pre-trained decoder head (usually 150 classes for ADE20k)
        # with a new binary head.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.MODEL_BACKBONE,
            num_labels=Config.NUM_CLASSES,
            num_channels=Config.IN_CHANNELS,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the SegFormer model.

        Args:
            x: Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            Logits tensor of shape (Batch, 1, Height, Width).
        """
        # SegFormer expects inputs as 'pixel_values'
        outputs = self.model(pixel_values=x)

        # The model outputs logits at 1/4th of the original resolution
        # Shape: (Batch, Num_Classes, H/4, W/4)
        logits = outputs.logits

        # Upsample logits to match the input image resolution
        # We use bilinear interpolation for smooth resizing
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

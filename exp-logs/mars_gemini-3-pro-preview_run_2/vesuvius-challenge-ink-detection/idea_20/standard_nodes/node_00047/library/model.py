import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class InkSegFormer(nn.Module):
    """
    SegFormer model wrapper for Vesuvius Ink Detection.

    Uses the MiT-B2 backbone (nvidia/mit-b2) with the standard MLP decoder.
    Designed to accept 3-channel inputs (Z-translated slabs) and output
    binary segmentation logits.
    """

    def __init__(self, model_name=Config.MODEL_NAME, num_labels=1):
        """
        Initialize the InkSegFormer.

        Args:
            model_name (str): HuggingFace model identifier (default: nvidia/mit-b2).
            num_labels (int): Number of output channels (default: 1 for binary).
        """
        super().__init__()

        # Load the SegFormer architecture.
        # We use 'ignore_mismatched_sizes=True' to allow replacing the pre-trained
        # classification head with our new custom head (num_labels=1).
        # The backbone weights (MiT-B2) are loaded from ImageNet pre-training.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=num_labels,
            num_channels=Config.IN_CHANNELS,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, num_labels, Height, Width).
        """
        # Pass input through the SegFormer backbone and decoder
        # Output is SemanticSegmenterOutput class
        outputs = self.model(pixel_values=x)

        # Logits are typically 1/4th of the input resolution (e.g., 128x128 for 512x512 input)
        logits = outputs.logits

        # Upsample logits to match the input image resolution
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

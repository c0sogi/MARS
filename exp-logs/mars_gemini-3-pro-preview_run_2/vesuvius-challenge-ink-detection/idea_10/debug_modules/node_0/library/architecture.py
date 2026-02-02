import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class SegFormerMiTB4(nn.Module):
    """
    SegFormer model with MiT-B4 backbone for binary segmentation.

    This implementation uses the Hugging Face Transformers library to load the
    SegFormer architecture. The backbone (MiT-B4) is initialized with ImageNet
    pretrained weights. The decoder is the standard SegFormer MLP decoder,
    adapted for binary output.
    """

    def __init__(self):
        super().__init__()

        # The Config defines ENCODER_NAME as "mit_b4".
        # We map this to the corresponding Hugging Face Hub identifier.
        # "nvidia/mit-b4" provides the MiT-B4 encoder weights pretrained on ImageNet.
        model_name = "nvidia/mit-b4"

        # Initialize the SegFormer model.
        # num_labels=Config.CLASSES (1) sets the output channels for binary segmentation.
        # ignore_mismatched_sizes=True allows loading the backbone weights
        # while initializing the new classification head from scratch.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name, num_labels=Config.CLASSES, ignore_mismatched_sizes=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).
                              Values should be normalized.

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1, Height, Width).
        """
        # The Hugging Face model expects the input under 'pixel_values'.
        # Input shape: (Batch, 3, H, W)
        outputs = self.model(pixel_values=x)

        # The model outputs logits at 1/4th the resolution of the input
        # (standard behavior for SegFormer due to the stride of the encoder/decoder).
        # Logits Shape: (Batch, 1, H/4, W/4)
        logits = outputs.logits

        # Upsample the logits to match the input resolution.
        # We use bilinear interpolation to recover the full spatial resolution.
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

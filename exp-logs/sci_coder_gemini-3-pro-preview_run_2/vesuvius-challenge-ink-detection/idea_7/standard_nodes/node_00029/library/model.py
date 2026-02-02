import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class OSSNet(nn.Module):
    """
    Overlapping Stratified SegFormer (OSS-Net) Wrapper.

    This class wraps the Hugging Face SegformerForSemanticSegmentation model
    to handle binary segmentation for the Vesuvius Ink Detection task.
    It ensures the output logits are upsampled to match the input resolution.
    """

    def __init__(self):
        super(OSSNet, self).__init__()

        # Load the SegFormer model with the specified backbone (MiT-B2)
        # We set num_labels=1 for binary classification (Ink vs No-Ink)
        # ignore_mismatched_sizes=True is required because we are loading
        # a backbone/classification checkpoint into a segmentation architecture,
        # so the decoder head weights will be initialized randomly.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.MODEL_ENCODER,
            num_labels=Config.NUM_CLASSES,
            num_channels=Config.IN_CHANNELS,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1, H, W).
        """
        # Pass input through the SegFormer model
        # The model expects 'pixel_values' as input
        outputs = self.model(pixel_values=x)

        # Extract logits.
        # SegFormer outputs logits at 1/4th of the original resolution (e.g., 128x128 for 512x512 input).
        logits = outputs.logits

        # Upsample logits to match the input image size
        # We use bilinear interpolation. align_corners=False is standard for segmentation.
        upsampled_logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False  # (H, W)
        )

        return upsampled_logits


def build_model():
    """
    Constructs and returns the OSS-Net model.

    Returns:
        nn.Module: The initialized PyTorch model.
    """
    model = OSSNet()
    return model

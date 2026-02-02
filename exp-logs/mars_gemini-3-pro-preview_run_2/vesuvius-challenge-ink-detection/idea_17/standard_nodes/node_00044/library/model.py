import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from library.config import Config


class VesuviusSegFormer(nn.Module):
    """
    SegFormer model wrapper for Vesuvius Ink Detection.
    Uses the MiT-B2 backbone with an MLP decoder.
    Upsamples the 1/4 resolution output of the MLP decoder to full input resolution.
    """

    def __init__(self):
        super().__init__()

        # Configure the model for Binary Segmentation (1 class)
        # We use the configuration from the pretrained backbone but override the label count.
        self.hf_config = SegformerConfig.from_pretrained(
            Config.MODEL_BACKBONE,
            num_labels=Config.NUM_CLASSES,
            reshape_last_stage=True,
        )

        # Initialize the model.
        # 'ignore_mismatched_sizes=True' is required because we are loading
        # an ImageNet-pretrained backbone (encoder) into a Segmentation architecture
        # (encoder+decoder), and changing the number of classes.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.MODEL_BACKBONE, config=self.hf_config, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1, H, W).
        """
        # Pass input through the SegFormer model
        # The model expects pixel_values argument
        outputs = self.model(pixel_values=x)

        # Extract logits. SegFormer outputs logits at 1/4th of the input resolution.
        # Shape: (Batch, Num_Classes, H/4, W/4)
        logits = outputs.logits

        # Upsample logits to match input resolution (H, W)
        # We use bilinear interpolation for smooth upscaling of the probability map.
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],  # Target size (H, W)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits


def build_segformer_model():
    """
    Factory function to build the VesuviusSegFormer model.
    """
    model = VesuviusSegFormer()
    return model

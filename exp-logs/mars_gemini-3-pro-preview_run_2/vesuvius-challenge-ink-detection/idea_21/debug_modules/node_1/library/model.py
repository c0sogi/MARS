import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class SpecialistSegFormer(nn.Module):
    """
    A SegFormer-based model specialized for binary ink detection on specific Z-depths.
    Wraps the Hugging Face SegformerForSemanticSegmentation class with an MiT-B2 backbone.
    """

    def __init__(self):
        """
        Initializes the SpecialistSegFormer model.

        Loads the MiT-B2 backbone with ImageNet-pretrained weights.
        Configures the All-MLP decoder for binary segmentation (1 class).
        """
        super(SpecialistSegFormer, self).__init__()

        # Map internal config encoder name to Hugging Face Hub model ID
        # Config.ENCODER_NAME is 'mit_b2', HF expects 'nvidia/mit-b2'
        model_name = f"nvidia/{Config.ENCODER_NAME.replace('_', '-')}"

        # Load the pretrained model
        # num_labels=1 sets the decoder to output a single channel (binary logits)
        # ignore_mismatched_sizes=True allows loading the encoder weights while
        # re-initializing the decoder head for our specific number of classes.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=Config.CLASSES,
            ignore_mismatched_sizes=True,
            num_channels=Config.IN_CHANNELS,
        )

    def forward(self, images):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input images tensor of shape (Batch, 3, Height, Width).
                                   Values should be normalized.

        Returns:
            torch.Tensor: Predicted logits of shape (Batch, 1, Height, Width).
        """
        # Pass input through the SegFormer model
        # The HF implementation expects 'pixel_values' argument
        outputs = self.model(pixel_values=images)

        # Extract logits from the output object
        # SegFormer outputs logits at 1/4th of the input resolution (H/4, W/4)
        logits = outputs.logits

        # Upsample logits to match the input image resolution
        # We use bilinear interpolation. align_corners=False is standard for segmentation.
        upsampled_logits = F.interpolate(
            logits,
            size=images.shape[-2:],  # Target size: (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

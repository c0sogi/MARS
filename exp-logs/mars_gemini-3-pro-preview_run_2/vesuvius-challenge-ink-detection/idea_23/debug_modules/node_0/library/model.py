import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from library.config import Config


class SegFormerSpecialist(nn.Module):
    """
    SegFormer Specialist Model for Ink Detection.

    Wraps the Hugging Face SegformerForSemanticSegmentation class.
    Uses the MiT-B2 backbone (~25M params) and the All-MLP decoder.

    Attributes:
        model (SegformerForSemanticSegmentation): The underlying HF model.
    """

    def __init__(self):
        """
        Initializes the SegFormer model with the MiT-B2 backbone.
        Configures the model for binary segmentation (num_labels=1).
        Loads ImageNet pretrained weights for the encoder.
        """
        super(SegFormerSpecialist, self).__init__()

        # Map local config backbone name to Hugging Face Hub identifier
        # Config.BACKBONE is 'mit_b2', HF hub expects 'nvidia/mit-b2'
        backbone_name = f"nvidia/{Config.BACKBONE.replace('_', '-')}"

        print(f"Initializing SegFormerSpecialist with backbone: {backbone_name}")

        # Configure for binary segmentation (Ink vs No-Ink)
        # 1 label results in a single channel output which we treat as logits for BCE
        self.config = SegformerConfig.from_pretrained(
            backbone_name, num_labels=1, reshape_last_stage=True
        )

        # Load the model with pretrained encoder weights.
        # The decoder head will be randomly initialized.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name, config=self.config, ignore_mismatched_sizes=True
        )

    def forward(self, images):
        """
        Forward pass of the network.

        Args:
            images (torch.Tensor): Input images of shape (Batch, 3, H, W).

        Returns:
            torch.Tensor: Upsampled logits of shape (Batch, 1, H, W).
        """
        # SegFormer expects 'pixel_values' argument
        outputs = self.model(pixel_values=images)

        # Extract logits. Shape is typically (Batch, NumLabels, H/4, W/4)
        logits = outputs.logits

        # Upsample logits to match input image resolution (512x512)
        # We use the shape of the input images to determine target size
        upsampled_logits = F.interpolate(
            logits,
            size=images.shape[-2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

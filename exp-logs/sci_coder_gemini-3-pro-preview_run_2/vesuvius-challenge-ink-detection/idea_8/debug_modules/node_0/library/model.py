import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, SegformerConfig
from library.config import BACKBONE, NUM_CLASSES


class InkSegFormer(nn.Module):
    """
    A wrapper around the HuggingFace SegFormer model with the MiT-B3 backbone.
    This class handles the initialization of the pretrained backbone and ensures
    the output logits are upsampled to the input image resolution.
    """

    def __init__(
        self, backbone_name=BACKBONE, num_classes=NUM_CLASSES, pretrained=True
    ):
        super(InkSegFormer, self).__init__()

        # Map internal config names to HuggingFace Hub identifiers
        # The task specifies nvidia/mit-b3
        if backbone_name == "mit_b3":
            self.hf_model_name = "nvidia/mit-b3"
        else:
            self.hf_model_name = backbone_name

        self.num_classes = num_classes

        # Configuration for the model
        # We explicitly set num_channels=3 to match the overlapping MIPs input strategy
        if pretrained:
            # Load pretrained encoder weights; decoder head is initialized randomly
            # ignore_mismatched_sizes is required because we are replacing the head
            self.model = SegformerForSemanticSegmentation.from_pretrained(
                self.hf_model_name,
                num_labels=self.num_classes,
                num_channels=3,
                ignore_mismatched_sizes=True,
            )
        else:
            # Initialize from scratch
            config = SegformerConfig.from_pretrained(
                self.hf_model_name, num_labels=self.num_classes, num_channels=3
            )
            self.model = SegformerForSemanticSegmentation(config)

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, Num_Classes, Height, Width).
        """
        # SegFormer expects input key 'pixel_values'
        outputs = self.model(pixel_values=x)

        # The logits from SegFormer are typically 1/4th of the input resolution
        logits = outputs.logits

        # Upsample logits to match input resolution
        # We use bilinear interpolation which is standard for segmentation masks
        upsampled_logits = F.interpolate(
            logits,
            size=x.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

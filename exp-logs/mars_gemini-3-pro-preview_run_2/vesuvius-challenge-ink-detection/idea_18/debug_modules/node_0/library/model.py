import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
import logging

# Import configuration
from library.config import Config

# Suppress warnings from transformers regarding uninitialized weights for the decoder head,
# which is expected behavior when fine-tuning a backbone on a new task.
logging.getLogger("transformers").setLevel(logging.ERROR)


class SegFormerB2(nn.Module):
    """
    SegFormer architecture using the MiT-B2 backbone and the standard MLP Decoder.

    This class wraps the Hugging Face SegformerForSemanticSegmentation model,
    initializing the encoder with ImageNet-pretrained weights ('nvidia/mit-b2')
    and adapting the head for binary segmentation (1 class).
    """

    def __init__(self):
        super(SegFormerB2, self).__init__()

        # Load the model with the MiT-B2 backbone.
        # ignore_mismatched_sizes=True is required because we are initializing
        # a segmentation model from a classification backbone (or changing the number of classes).
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            "nvidia/mit-b2",
            num_labels=Config.NUM_CLASSES,
            ignore_mismatched_sizes=True,
            id2label={0: "background", 1: "ink"},
            label2id={"background": 0, "ink": 1},
        )

    def forward(self, images):
        """
        Forward pass of the SegFormer model.

        Args:
            images (torch.Tensor): Input tensor of shape (Batch_Size, 3, Height, Width).
                                   Values should be normalized.

        Returns:
            torch.Tensor: Upsampled logits of shape (Batch_Size, 1, Height, Width).
        """
        # Pass inputs to the HF model
        # The model outputs a SemanticSegmenterOutput object
        outputs = self.model(pixel_values=images)

        # Retrieve logits.
        # By default, SegFormer outputs logits at 1/4th the resolution of the input.
        # Shape: (Batch_Size, Num_Classes, H/4, W/4)
        logits = outputs.logits

        # Upsample logits to the original input resolution (512x512)
        # Bilinear interpolation is standard for segmentation masks.
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=images.shape[-2:],  # Target size: (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

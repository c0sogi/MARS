import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class SpecialistModel(nn.Module):
    """
    A wrapper around the SegFormer architecture for ink detection.
    Uses the MiT-B2 backbone with the standard All-MLP decoder.
    """

    def __init__(self, model_name="nvidia/mit-b2"):
        """
        Initializes the SegFormer model.

        Args:
            model_name (str): The HuggingFace checkpoint name for the backbone.
                              Defaults to 'nvidia/mit-b2'.
        """
        super().__init__()

        # Load the SegFormer model with a custom number of classes (1 for binary ink detection).
        # ignore_mismatched_sizes=True is required because we are loading encoder weights ('nvidia/mit-b2')
        # into a full segmentation architecture, or replacing the head of a fine-tuned model.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=Config.NUM_CLASSES,
            id2label={0: "ink"},
            label2id={"ink": 0},
            ignore_mismatched_sizes=True,
        )

    def forward(self, images):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Logits of shape (Batch, 1, Height, Width).
        """
        # SegFormer expects pixel_values argument
        outputs = self.model(pixel_values=images)

        # The raw logits from SegFormer are typically 1/4th of the input resolution
        logits = outputs.logits

        # Upsample logits to match the input image size
        # We use bilinear interpolation for the logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=images.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

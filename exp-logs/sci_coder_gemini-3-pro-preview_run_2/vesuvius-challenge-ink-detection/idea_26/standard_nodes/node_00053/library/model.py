import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class InkSegFormer(nn.Module):
    """
    SegFormer MiT-B2 model with the standard MLP Decoder for binary segmentation.

    This class wraps the Hugging Face SegformerForSemanticSegmentation implementation.
    It loads ImageNet-pretrained weights for the MiT-B2 backbone and initializes
    a fresh MLP decoder for the binary ink detection task.
    """

    def __init__(self):
        super(InkSegFormer, self).__init__()

        # Determine the Hugging Face model ID based on the config
        # Defaulting to nvidia/mit-b2 as specified in the task description
        if Config.ENCODER_NAME == "mit_b2":
            model_id = "nvidia/mit-b2"
        else:
            # Fallback if config changes, though task specifies mit_b2
            model_id = "nvidia/mit-b2"

        # Initialize the SegFormer model
        # num_labels=1 configures the MLP decoder for binary output.
        # ignore_mismatched_sizes=True allows loading the encoder weights (mit-b2)
        # into the full segmentation architecture (encoder + new decoder).
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            model_id, num_labels=Config.CLASSES, ignore_mismatched_sizes=True
        )

    def forward(self, x):
        """
        Forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).

        Returns:
            torch.Tensor: Output logits of shape (Batch, 1, Height, Width).
        """
        # Pass input through the SegFormer model
        # The model expects the argument 'pixel_values'
        outputs = self.model(pixel_values=x)

        # The Hugging Face SegFormer implementation outputs logits at 1/4th the resolution
        # of the input image (stride 4).
        logits = outputs.logits

        # Upsample the logits to match the input spatial dimensions (512x512)
        # Bilinear interpolation is standard for segmentation mask upsampling.
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=x.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

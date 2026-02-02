import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

from library.config import MODEL_PARAMS


class VesuviusSegFormer(nn.Module):
    """
    SegFormer model wrapper for Vesuvius Ink Detection.
    Uses the MiT-B2 backbone with an All-MLP decoder as defined in the SegFormer paper.
    """

    def __init__(self, model_name, num_classes=1, in_channels=3):
        super().__init__()

        # Map generic configuration names to Hugging Face Hub IDs
        name_map = {
            "mit_b0": "nvidia/mit-b0",
            "mit_b1": "nvidia/mit-b1",
            "mit_b2": "nvidia/mit-b2",
            "mit_b3": "nvidia/mit-b3",
            "mit_b4": "nvidia/mit-b4",
            "mit_b5": "nvidia/mit-b5",
        }
        hf_model_name = name_map.get(model_name, model_name)

        # Load the model with a fresh decoder head for our specific number of classes.
        # ignore_mismatched_sizes=True is crucial here: it allows loading the
        # ImageNet-pretrained encoder weights while ignoring the mismatch in the
        # decoder head (which we are replacing/resizing for binary classification).
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            hf_model_name,
            num_labels=num_classes,
            num_channels=in_channels,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (Batch, 3, H, W)
        Returns:
            logits: Output tensor of shape (Batch, num_classes, H, W)
        """
        # Pass input to the Hugging Face model
        outputs = self.model(pixel_values=x)

        # Extract logits.
        # Note: SegFormer outputs logits at 1/4th of the original resolution (e.g., 128x128 for 512x512 input).
        logits = outputs.logits

        # Upsample logits to match the input resolution
        upsampled_logits = F.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )

        return upsampled_logits


def get_model(config=None):
    """
    Factory function to create the model based on configuration.

    Args:
        config: Dictionary containing model hyperparameters.
                Defaults to library.config.MODEL_PARAMS.

    Returns:
        Initialized PyTorch model.
    """
    if config is None:
        config = MODEL_PARAMS

    # Extract parameters with defaults
    encoder_name = config.get("encoder_name", "mit_b2")
    classes = config.get("classes", 1)
    in_channels = config.get("in_channels", 3)

    # Instantiate the wrapper
    model = VesuviusSegFormer(
        model_name=encoder_name, num_classes=classes, in_channels=in_channels
    )

    return model

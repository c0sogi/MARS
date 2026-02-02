import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation
from library.config import Config


class SegFormerMiTB2(nn.Module):
    """
    Translation-Invariant SegFormer (MiT-B2) for Ink Detection.

    This model utilizes the MiT-B2 backbone pre-trained on ImageNet, coupled with
    the standard MLP decoder provided by the SegFormer architecture. It is adapted
    for binary segmentation (Ink vs No-Ink).

    Attributes:
        segformer (SegformerForSemanticSegmentation): The underlying HF model.
    """

    def __init__(self):
        super(SegFormerMiTB2, self).__init__()

        # Ensure reproducibility for initialization
        torch.manual_seed(Config.SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(Config.SEED)

        # Load the pretrained SegFormer B2 model
        # We use 'nvidia/mit-b2' which corresponds to the Mix Transformer B2 encoder.
        # The SegformerForSemanticSegmentation class includes the All-MLP decoder.
        # We set num_labels=1 for binary classification.
        # ignore_mismatched_sizes=True allows us to replace the 1000-class ImageNet head
        # with our 1-class head.
        self.model_name = "nvidia/mit-b2"
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            self.model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
            reshape_last_stage=True,
        )

    def forward(self, pixel_values):
        """
        Forward pass of the model.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape (Batch, 3, Height, Width).
                                         Values should be normalized to [0, 1].

        Returns:
            torch.Tensor: Logits of shape (Batch, 1, Height, Width).
        """
        # Pass inputs through the SegFormer model
        # Output is a SemanticSegmenterOutput object containing 'logits'
        outputs = self.segformer(pixel_values=pixel_values)

        # Extract logits
        # Shape is typically (Batch, num_labels, H/4, W/4) due to the architecture's stride
        logits = outputs.logits

        # Upsample logits to match the input resolution (512x512)
        # We use bilinear interpolation.
        # align_corners=False is standard for segmentation tasks.
        upsampled_logits = F.interpolate(
            logits,
            size=pixel_values.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        return upsampled_logits

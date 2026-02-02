import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation, logging
from library.config import Config

# Suppress warnings related to weight initialization when loading the backbone
logging.set_verbosity_error()


class SegFormerB2(nn.Module):
    """
    SegFormer MiT-B2 model with an MLP Decoder for Binary Segmentation.
    Wraps the Hugging Face transformers implementation.
    """

    def __init__(self):
        super(SegFormerB2, self).__init__()

        # Load the pretrained SegFormer model
        # We use the 'nvidia/mit-b2' checkpoint which contains the MixVisionTransformer backbone.
        # We override the decoder hidden size and number of labels to match our specific task.
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            Config.ENCODER_NAME,  # "nvidia/mit-b2"
            num_labels=Config.CLASSES,
            num_channels=Config.NUM_CHANNELS,
            decoder_hidden_size=Config.DECODER_DIM,
            ignore_mismatched_sizes=True,
        )

        # Initialize Binary Cross Entropy Loss
        self.bce_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, images, labels=None):
        """
        Forward pass of the model.

        Args:
            images (torch.Tensor): Input tensor of shape (Batch, Channels, Height, Width).
            labels (torch.Tensor, optional): Target binary masks of shape (Batch, 1, Height, Width).

        Returns:
            dict: A dictionary containing:
                - "logits": The upsampled prediction logits (Batch, 1, Height, Width).
                - "loss": The scalar loss value (if labels are provided).
        """
        # Pass input through the SegFormer model
        # The model outputs a SemanticSegmenterOutput object
        outputs = self.model(pixel_values=images)

        # Extract logits. SegFormer outputs logits at 1/4th of the input resolution (e.g., 128x128 for 512x512 input)
        logits = outputs.logits

        # Upsample logits to match the input image resolution
        upsampled_logits = F.interpolate(
            logits,
            size=images.shape[-2:],  # (Height, Width)
            mode="bilinear",
            align_corners=False,
        )

        result = {"logits": upsampled_logits}

        # Calculate Loss if labels are provided
        if labels is not None:
            # 1. Binary Cross Entropy Loss
            bce_loss = self.bce_loss_fn(upsampled_logits, labels)

            # 2. Dice Loss
            # Apply sigmoid to convert logits to probabilities [0, 1]
            probs = torch.sigmoid(upsampled_logits)

            # Flatten tensors to calculate Dice over the batch or image
            # Here we flatten the entire batch to treat it as a global volume for stability
            probs_flat = probs.view(-1)
            labels_flat = labels.view(-1)

            intersection = (probs_flat * labels_flat).sum()
            union = probs_flat.sum() + labels_flat.sum()

            # Add epsilon to avoid division by zero
            epsilon = 1e-7
            dice_score = (2.0 * intersection) / (union + epsilon)
            dice_loss = 1.0 - dice_score

            # 3. Combined Loss
            total_loss = (Config.BCE_WEIGHT * bce_loss) + (
                Config.DICE_WEIGHT * dice_loss
            )

            result["loss"] = total_loss

        return result

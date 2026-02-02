import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation
from library.config import Config


def get_model():
    """
    Constructs and returns the Fine-Grained Stratified SegFormer model.

    This function:
    1. Loads the SegformerForSemanticSegmentation model with the MiT-B2 backbone.
    2. Modifies the first patch embedding layer to accept Config.IN_CHANNELS (6) instead of 3.
    3. Initializes the new channels by replicating and scaling the pre-trained RGB weights
       to preserve feature distribution statistics.

    Returns:
        torch.nn.Module: The modified SegFormer model.
    """
    # Load pre-trained model
    # ignore_mismatched_sizes is True because we are likely initializing a segmentation head
    # with a different number of classes (1) than the pre-training objective.
    model = SegformerForSemanticSegmentation.from_pretrained(
        Config.BACKBONE,
        num_labels=Config.NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # Access the first patch embedding layer in the encoder
    # Structure: model -> segformer -> encoder -> patch_embeddings -> [0] -> proj
    old_proj = model.segformer.encoder.patch_embeddings[0].proj

    # Check if modification is necessary
    if old_proj.in_channels != Config.IN_CHANNELS:
        # Create a new convolution layer with the target number of input channels
        new_proj = nn.Conv2d(
            in_channels=Config.IN_CHANNELS,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=(old_proj.bias is not None),
        )

        # Initialize weights
        # We assume the input channels are groups of RGB-like features or simply repeated modalities.
        # To preserve the magnitude of activations, we scale the weights.
        # Scale factor = Old Channels / New Channels (e.g., 3 / 6 = 0.5)
        # This assumes the new channels contribute additively to the activation.
        with torch.no_grad():
            # Get original weights: (Out, 3, K, K)
            old_weights = old_proj.weight

            # Create a tensor for new weights
            # We repeat the 3-channel weights to fill the 6 channels
            # For 6 channels, we repeat 2 times (6 // 3)
            repeat_factor = Config.IN_CHANNELS // old_proj.in_channels
            if Config.IN_CHANNELS % old_proj.in_channels != 0:
                # Fallback for non-integer multiples: just repeat and slice
                # (Not strictly needed for 3->6 case but good for robustness)
                temp_weights = old_weights.repeat(1, repeat_factor + 1, 1, 1)
                new_weights = temp_weights[:, : Config.IN_CHANNELS, :, :]
            else:
                new_weights = old_weights.repeat(1, repeat_factor, 1, 1)

            # Scale weights
            # If we sum 6 channels instead of 3, the output magnitude doubles.
            # We multiply by (3/6) = 0.5 to normalize.
            scale = old_proj.in_channels / Config.IN_CHANNELS
            new_weights = new_weights * scale

            # Assign weights
            new_proj.weight.copy_(new_weights)

            # Copy bias if it exists
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)

        # Replace the layer in the model
        model.segformer.encoder.patch_embeddings[0].proj = new_proj

        # Update config to reflect the change (good practice for saving/loading later)
        model.config.num_channels = Config.IN_CHANNELS

        print(
            f"Model modified: Input channels expanded from {old_proj.in_channels} to {Config.IN_CHANNELS}."
        )
        print(
            f"Weights scaled by factor {scale:.4f} to preserve activation distribution."
        )

    return model

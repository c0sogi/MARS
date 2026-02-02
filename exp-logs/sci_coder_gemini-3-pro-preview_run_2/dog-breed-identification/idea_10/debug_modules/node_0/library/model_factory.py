import torch
import torch.nn as nn
from torchvision import models
from library.config import Config


def get_backbone(stream_name):
    """
    Initializes and modifies the backbone model for the specified stream.
    Loads pre-trained weights and modifies the architecture to return
    feature embeddings instead of class logits.

    Args:
        stream_name (str): 'stream_a' (ConvNeXt) or 'stream_b' (ViT).

    Returns:
        torch.nn.Module: The modified backbone model in eval mode with frozen parameters.
    """
    model = None

    if stream_name == "stream_a":
        # Stream A: ConvNeXt Large
        # Weights: IMAGENET1K_V1 (Torchvision 'New Recipe')
        print(f"Loading {Config.MODEL_A_NAME} with weights {Config.MODEL_A_WEIGHTS}...")

        try:
            model = models.convnext_large(weights=Config.MODEL_A_WEIGHTS)
        except Exception as e:
            raise RuntimeError(f"Failed to load ConvNeXt model: {e}")

        # Modify to return embeddings (Post-Pooling/LayerNorm)
        # ConvNeXt classifier structure in torchvision:
        # (0): LayerNorm2d
        # (1): Flatten
        # (2): Linear (Head)
        # We replace the Linear layer with Identity to keep the LayerNorm and Flatten operations.
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            # Verify the last layer is Linear before replacing
            if isinstance(model.classifier[-1], nn.Linear):
                model.classifier[-1] = nn.Identity()
            else:
                # Fallback if structure is slightly different but sequential
                # Assuming the last layer is the classification head
                model.classifier[-1] = nn.Identity()
        else:
            raise ValueError(
                "Unexpected ConvNeXt model structure: 'classifier' not found or not Sequential."
            )

    elif stream_name == "stream_b":
        # Stream B: ViT Large 16
        # Weights: IMAGENET1K_SWAG_E2E_V1 (SWAG Weights)
        print(f"Loading {Config.MODEL_B_NAME} with weights {Config.MODEL_B_WEIGHTS}...")

        try:
            model = models.vit_l_16(weights=Config.MODEL_B_WEIGHTS)
        except Exception as e:
            raise RuntimeError(f"Failed to load ViT model: {e}")

        # Modify to return embeddings (Post-Pooling/LayerNorm)
        # ViT structure in torchvision:
        # self.encoder (includes final LayerNorm)
        # self.heads = nn.Sequential(nn.Linear)
        # The forward pass extracts the CLS token and passes it to self.heads.
        # Replacing self.heads with Identity returns the CLS token embedding.
        if hasattr(model, "heads"):
            model.heads = nn.Identity()
        else:
            raise ValueError(
                "Unexpected ViT model structure: 'heads' attribute not found."
            )

    else:
        raise ValueError(
            f"Unknown stream name: {stream_name}. Expected 'stream_a' or 'stream_b'."
        )

    # Freeze all parameters for Feature Extraction
    # This saves memory and ensures the backbone is treated as a fixed feature extractor.
    for param in model.parameters():
        param.requires_grad = False

    # Set to evaluation mode (disables Dropout, uses running stats for BatchNorm if present)
    model.eval()

    return model

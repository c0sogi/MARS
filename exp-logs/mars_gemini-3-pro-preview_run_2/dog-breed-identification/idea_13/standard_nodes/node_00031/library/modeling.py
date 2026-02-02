import torch
import timm
import os
from library.config import Config


def load_feature_extractor(model_name: str, device: str = Config.DEVICE):
    """
    Initializes a timm model as a fixed feature extractor.

    This function loads a pretrained model, removes its classification head
    to expose the pooled feature embeddings, and freezes all parameters.

    Args:
        model_name (str): The name of the model architecture to load (e.g., 'convnext_large', 'vit_large_patch14_dinov2').
        device (str): The device to load the model onto ('cpu' or 'cuda').

    Returns:
        torch.nn.Module: The initialized model with the classification head removed,
                         set to eval mode, and gradients disabled.
    """
    # Create the model with pretrained weights
    # We rely on timm to download and cache the weights automatically.
    try:
        # Cite debug_lesson_2: Synchronize Configuration Definitions.
        # The DINOv2 model defaults to 518x518, but the data pipeline provides 224x224 images.
        # We explicitly set img_size=224 for DINOv2 to force position embedding interpolation.
        kwargs = {}
        if "dinov2" in model_name:
            kwargs["img_size"] = 224

        model = timm.create_model(model_name, pretrained=True, **kwargs)
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}'. Error: {e}")

    # Remove the classification head to get pooled features.
    # reset_classifier(0) replaces the head with Identity (or equivalent for the architecture)
    # and sets num_classes to 0. This ensures we get the embedding after the final global pooling
    # and layer normalization.
    model.reset_classifier(0)

    # Move the model to the specified computation device
    model = model.to(device)

    # Set the model to evaluation mode
    # This fixes layers like Dropout and BatchNorm to their inference states.
    model.eval()

    # Disable gradients for all parameters
    # This reduces memory usage and ensures the backbone remains fixed (no fine-tuning).
    for param in model.parameters():
        param.requires_grad = False

    return model

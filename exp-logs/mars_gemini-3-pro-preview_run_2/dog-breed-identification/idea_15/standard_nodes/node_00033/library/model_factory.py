import torch
import torch.nn as nn
import torchvision
import timm
import sys
from library.config import DEVICE


def load_backbone(stream_config):
    """
    Loads the backbone model based on the stream configuration.
    Configures the model for feature extraction (removes classification head,
    freezes weights, sets to eval mode).

    Args:
        stream_config (dict): Configuration dictionary for the stream.

    Returns:
        torch.nn.Module: The configured backbone model on the correct device.
    """
    library = stream_config.get("library")
    model_name = stream_config.get("model_name")

    print(f"Loading backbone for {stream_config['name']}: {model_name} ({library})")

    if library == "torchvision":
        # Load weights
        weights_enum = stream_config.get("weights")
        # Resolve string to actual weights object if necessary, or pass string if supported
        # torchvision >= 0.13 supports strings like "IMAGENET1K_V1"

        try:
            model = torchvision.models.convnext_large(weights=weights_enum)
        except Exception as e:
            print(f"Error loading torchvision model: {e}")
            raise

        # Modify for feature extraction
        # ConvNeXt classifier block is:
        # (0): LayerNorm2d((1536,), eps=1e-06, elementwise_affine=True)
        # (1): Flatten(start_dim=1, end_dim=-1)
        # (2): Linear(in_features=1536, out_features=1000, bias=True)
        # We want to keep LayerNorm and Flatten, but remove Linear.
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            # Replace the last layer (Linear) with Identity
            model.classifier[-1] = nn.Identity()
        else:
            raise ValueError(
                "Unexpected ConvNeXt architecture: classifier is not a Sequential block or missing."
            )

    elif library == "timm":
        pretrained = stream_config.get("pretrained", True)
        try:
            # num_classes=0 removes the final linear layer
            # global_pool='' allows us to control pooling, but usually default is fine for feature extraction
            # MaxViT in timm with num_classes=0 returns the pooled feature vector
            model = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        except Exception as e:
            print(f"Error loading timm model: {e}")
            raise

    else:
        raise ValueError(f"Unsupported library: {library}")

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Move to device and set to eval mode
    model = model.to(DEVICE)
    model.eval()

    return model

import torch
import torch.nn as nn
import torchvision
import timm
from library.config import Config


def load_backbone(stream_config):
    """
    Factory function to load backbone models for the Dual-Stream Heterogeneous Ensemble.

    Args:
        stream_config (dict): Configuration dictionary (e.g., Config.STREAM_A or Config.STREAM_B)
                              containing 'name', 'library', and 'weights'.

    Returns:
        nn.Module: The loaded backbone model with the classification head replaced by Identity.
                   The model is placed on the configured device and set to eval mode.
    """
    name = stream_config["name"]
    library = stream_config["library"]

    # print(f"Loading backbone: {name} from {library}...")

    if library == "torchvision":
        if name == "convnext_large":
            # Load ConvNeXt Large with default ImageNet1K V1 weights
            # Config implies 'DEFAULT' weights
            weights = torchvision.models.ConvNeXt_Large_Weights.DEFAULT
            model = torchvision.models.convnext_large(weights=weights)

            # The ConvNeXt classifier block in torchvision is:
            # Sequential(
            #   (0): LayerNorm2d(...)
            #   (1): Flatten(...)
            #   (2): Linear(...)
            # )
            # We must preserve the LayerNorm and Flatten layers for correct feature extraction.
            # We replace only the final Linear layer with Identity.
            if hasattr(model, "classifier") and isinstance(
                model.classifier, nn.Sequential
            ):
                model.classifier[-1] = nn.Identity()
            else:
                # Fallback in case of unexpected architecture changes
                model.classifier = nn.Identity()

        else:
            raise NotImplementedError(
                f"Torchvision model '{name}' is not supported in this factory."
            )

    elif library == "timm":
        # Load model using timm (PyTorch Image Models)
        # Setting num_classes=0 is the standard way to remove the classification head
        # while retaining the final normalization (LayerNorm) and pooling layers.
        # This is critical for ViT/EVA02 models to output correct embeddings.
        try:
            model = timm.create_model(name, pretrained=True, num_classes=0)
        except Exception as e:
            raise RuntimeError(f"Failed to load timm model '{name}': {e}")

    else:
        raise ValueError(
            f"Library '{library}' is not supported. Use 'torchvision' or 'timm'."
        )

    # Move model to the computation device (GPU/CPU)
    model = model.to(Config.DEVICE)

    # Set to evaluation mode to freeze BatchNorm stats and disable Dropout
    model.eval()

    return model

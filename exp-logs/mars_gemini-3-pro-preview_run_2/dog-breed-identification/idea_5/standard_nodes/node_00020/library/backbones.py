import torch
import torch.nn as nn
from torchvision.models import get_model
import library.config as config


def load_feature_extractor(model_name, weights_name, device=config.DEVICE, freeze=True):
    """
    Loads a torchvision model, loads the specified weights, removes the classification head,
    and optionally freezes the parameters to serve as a feature extractor.

    Args:
        model_name (str): Name of the model architecture (e.g., 'convnext_large').
        weights_name (str): Name of the pretrained weights (e.g., 'IMAGENET1K_V1').
        device (str): Device to load the model onto ('cpu' or 'cuda').
        freeze (bool): If True, sets requires_grad=False for all parameters.

    Returns:
        model (nn.Module): The modified model returning raw embeddings.
    """
    print(f"Initializing feature extractor: {model_name} with weights: {weights_name}")

    try:
        # Load the model with the specified weights
        model = get_model(model_name, weights=weights_name)
    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        raise e

    # Architecture-specific modifications to remove classification head
    if "convnext" in model_name.lower():
        # ConvNeXt structure: features -> avgpool -> classifier
        # classifier is Sequential(LayerNorm2d, Flatten, Linear)
        # We replace the Linear layer (index 2) with Identity to keep LayerNorm and Flatten
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            if len(model.classifier) >= 3 and isinstance(
                model.classifier[2], nn.Linear
            ):
                input_dim = model.classifier[2].in_features
                model.classifier[2] = nn.Identity()
                print(
                    f"  -> Modified ConvNeXt classifier: Replaced Linear(in={input_dim}) with Identity."
                )
            else:
                print(
                    f"  -> Warning: ConvNeXt classifier structure unexpected. Expected Linear at index 2."
                )
        else:
            print(
                f"  -> Warning: 'classifier' attribute not found or not Sequential in ConvNeXt model."
            )

    elif "vit" in model_name.lower():
        # ViT structure: encoder -> heads
        # heads is Sequential(Linear)
        # We replace the entire heads block with Identity
        if hasattr(model, "heads") and isinstance(model.heads, nn.Sequential):
            model.heads = nn.Identity()
            print(f"  -> Modified ViT heads: Replaced Sequential block with Identity.")
        elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
            # Fallback for some ViT variants
            model.head = nn.Identity()
            print(f"  -> Modified ViT head: Replaced Linear layer with Identity.")
        else:
            print(f"  -> Warning: 'heads' or 'head' attribute not found in ViT model.")

    elif "resnet" in model_name.lower():
        # Fallback for ResNet-like architectures (fc layer)
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            model.fc = nn.Identity()
            print(f"  -> Modified ResNet fc: Replaced Linear layer with Identity.")

    elif "efficientnet" in model_name.lower():
        # EfficientNet: classifier is Sequential(Dropout, Linear)
        if hasattr(model, "classifier") and isinstance(model.classifier, nn.Sequential):
            model.classifier[-1] = nn.Identity()
            print(
                f"  -> Modified EfficientNet classifier: Replaced last layer with Identity."
            )

    else:
        print(
            f"  -> Warning: Unknown architecture '{model_name}'. No classification head removed."
        )

    # Freeze parameters
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        print(f"  -> Backbone parameters frozen.")

    # Move to device and set to eval mode
    model = model.to(device)
    model.eval()

    return model

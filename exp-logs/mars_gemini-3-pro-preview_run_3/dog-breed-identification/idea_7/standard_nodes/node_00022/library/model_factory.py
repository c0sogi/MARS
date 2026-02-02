import torch
import torch.nn as nn
import timm
from library.config import Config


def create_model(model_name, num_classes=Config.NUM_CLASSES, pretrained=True):
    """
    Creates a model using the timm library.

    Args:
        model_name (str): Name of the model architecture (must be supported by timm).
        num_classes (int): Number of output classes for the classification head.
        pretrained (bool): Whether to load pretrained ImageNet weights.

    Returns:
        model (nn.Module): The instantiated PyTorch model.
    """
    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to create model {model_name}: {e}")


def get_llrd_params(
    model,
    model_name,
    lr=Config.LEARNING_RATE,
    weight_decay=Config.WEIGHT_DECAY,
    layer_decay=Config.LLRD_DECAY,
):
    """
    Groups model parameters for Layer-Wise Learning Rate Decay (LLRD).

    Divides the model into layers (Stem, Stages 0-3, Head) and applies a decaying
    learning rate: lr * (layer_decay ** depth_from_head).
    Also separates parameters for weight decay (0 for biases/normalization).

    Args:
        model (nn.Module): The model to optimize.
        model_name (str): Name of the model to determine architecture specific grouping.
        lr (float): Base learning rate (applied to the head).
        weight_decay (float): Weight decay coefficient.
        layer_decay (float): Decay rate per layer (0.0 < decay <= 1.0).

    Returns:
        list[dict]: A list of parameter groups compatible with PyTorch optimizers.
    """

    # Define groupings based on architecture
    # We assign a 'layer_id' to each parameter.
    # Higher ID = Closer to output (Higher LR).
    # ID 5: Head / Final Norm
    # ID 4: Stage 3 / Layer 3
    # ID 3: Stage 2 / Layer 2
    # ID 2: Stage 1 / Layer 1
    # ID 1: Stage 0 / Layer 0
    # ID 0: Stem / Embeddings

    is_swin = "swin" in model_name

    # Dictionary to hold params: key=(layer_id, apply_weight_decay) -> value=[params]
    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Layer ID
        layer_id = 0  # Default to stem/lowest

        if is_swin:
            if "head" in name or name.startswith("norm."):
                layer_id = 5
            elif "layers.3" in name:
                layer_id = 4
            elif "layers.2" in name:
                layer_id = 3
            elif "layers.1" in name:
                layer_id = 2
            elif "layers.0" in name:
                layer_id = 1
            else:
                # patch_embed, absolute_pos_embed, etc.
                layer_id = 0
        else:  # ConvNeXt and generic fallbacks
            if "head" in name or name.startswith("norm."):
                layer_id = 5
            elif "stages.3" in name:
                layer_id = 4
            elif "stages.2" in name:
                layer_id = 3
            elif "stages.1" in name:
                layer_id = 2
            elif "stages.0" in name:
                layer_id = 1
            else:
                # stem, etc.
                layer_id = 0

        # 2. Determine Weight Decay application
        # Apply WD to weights (len > 1), but NOT to biases or Norm params
        # Heuristic: if name contains 'bias' or 'norm' or 'bn' -> no WD
        # Also check dimensionality (1D usually means bias/scale)
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name or "bn" in name:
            apply_wd = False
        else:
            apply_wd = True

        # Add to group
        key = (layer_id, apply_wd)
        if key not in param_groups:
            param_groups[key] = []
        param_groups[key].append(param)

    # 3. Construct Optimizer Param Groups
    final_groups = []

    # Iterate through possible keys
    # layer_ids 0 to 5
    for layer_id in range(6):
        # Calculate LR for this layer
        # Head (5) gets lr
        # Layer 4 gets lr * decay
        # ...
        # Layer 0 gets lr * decay^5

        scale = layer_decay ** (5 - layer_id)
        scaled_lr = lr * scale

        # Add group with WD
        if (layer_id, True) in param_groups:
            final_groups.append(
                {
                    "params": param_groups[(layer_id, True)],
                    "lr": scaled_lr,
                    "weight_decay": weight_decay,
                }
            )

        # Add group without WD
        if (layer_id, False) in param_groups:
            final_groups.append(
                {
                    "params": param_groups[(layer_id, False)],
                    "lr": scaled_lr,
                    "weight_decay": 0.0,
                }
            )

    return final_groups

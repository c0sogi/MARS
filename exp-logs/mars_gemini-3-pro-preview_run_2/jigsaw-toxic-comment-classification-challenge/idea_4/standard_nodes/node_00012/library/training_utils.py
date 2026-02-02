import re
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


def get_llrd_optimizer_params(model, lr_backbone, lr_head, weight_decay, llrd_decay):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    This function groups model parameters to apply different learning rates based on their
    depth in the network. The classification head gets the highest LR, while the backbone
    layers decay geometrically as they get closer to the input.

    Args:
        model: The CustomTransformer model instance.
        lr_backbone (float): Base learning rate for the top layer of the transformer backbone.
        lr_head (float): Learning rate for the classification head.
        weight_decay (float): Weight decay coefficient for regularization.
        llrd_decay (float): Multiplicative decay factor (0 < decay <= 1) for lower layers.

    Returns:
        list: A list of dictionaries containing parameter groups, suitable for torch.optim.Optimizer.
    """

    # Define parameters to exclude from weight decay (biases and normalization terms)
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Container for grouping parameters by (lr, weight_decay)
    # Key: (lr, weight_decay), Value: list of parameters
    optimizer_grouped_parameters = {}

    # Determine the number of layers in the backbone
    # CustomTransformer wraps the HF model in self.backbone, so we check its config
    if hasattr(model, "config") and hasattr(model.config, "num_hidden_layers"):
        num_layers = model.config.num_hidden_layers
    else:
        # Fallback: Infer number of layers from parameter names if config is not directly accessible
        layers = [
            int(re.search(r"layer\.(\d+)\.", n).group(1))
            for n, _ in model.named_parameters()
            if "layer." in n and re.search(r"layer\.(\d+)\.", n)
        ]
        num_layers = (
            max(layers) + 1 if layers else 12
        )  # Default to 12 (Base) if detection fails

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Weight Decay
        # Exclude bias and LayerNorm from weight decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # 2. Determine Learning Rate
        if "head" in name:
            # Classification head gets a specific (usually higher) LR
            lr = lr_head
        elif "embeddings" in name:
            # Embeddings are at the very bottom, so they get the most decay
            lr = lr_backbone * (llrd_decay**num_layers)
        else:
            # Check if the parameter belongs to a specific encoder layer
            # Regex looks for patterns like 'encoder.layer.0.', 'layers.11.', etc.
            match = re.search(r"layer\.(\d+)\.", name)
            if match:
                layer_idx = int(match.group(1))
                # Apply decay based on depth.
                # Top layer (index = num_layers - 1) gets lr_backbone * decay^0
                # Bottom layer (index = 0) gets lr_backbone * decay^(num_layers - 1)
                lr = lr_backbone * (llrd_decay ** (num_layers - 1 - layer_idx))
            else:
                # Other backbone parameters (e.g., pooler, final layernorm)
                # Usually kept at the base backbone LR
                lr = lr_backbone

        # 3. Group parameters
        # We use the (lr, wd) tuple as a key to aggregate parameters
        key = (lr, wd)
        if key not in optimizer_grouped_parameters:
            optimizer_grouped_parameters[key] = []
        optimizer_grouped_parameters[key].append(param)

    # Convert the dictionary groups to the list format required by PyTorch optimizers
    params_list = []
    for (lr, wd), params in optimizer_grouped_parameters.items():
        params_list.append({"params": params, "lr": lr, "weight_decay": wd})

    return params_list


def get_scheduler(optimizer, num_train_steps, warmup_ratio):
    """
    Creates a Cosine Schedule with Warmup.

    Args:
        optimizer (torch.optim.Optimizer): The optimizer for which to schedule the learning rate.
        num_train_steps (int): Total number of training steps.
        warmup_ratio (float): The fraction of total steps to use for the linear warmup phase.

    Returns:
        torch.optim.lr_scheduler.LambdaLR: The HuggingFace learning rate scheduler.
    """
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    return scheduler

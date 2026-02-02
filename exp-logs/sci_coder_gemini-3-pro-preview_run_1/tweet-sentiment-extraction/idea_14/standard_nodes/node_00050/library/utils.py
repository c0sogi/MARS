import os
import random
import numpy as np
import torch
import re


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.

    Args:
        seed (int): The seed value to use.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_text(text):
    """
    Normalizes text by collapsing multiple spaces and stripping leading/trailing whitespace.
    This ensures consistency between raw text and tokenized input, adhering to the
    'Normalize-First' protocol.

    Args:
        text (str): The input text string.

    Returns:
        str: The normalized text.
    """
    if not isinstance(text, str):
        return str(text)
    # Replace multiple whitespaces with a single space
    text = re.sub(r"\s+", " ", text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def jaccard(str1, str2):
    """
    Computes the word-level Jaccard similarity score between two strings.

    Args:
        str1 (str): First string.
        str2 (str): Second string.

    Returns:
        float: The Jaccard score (intersection over union of words).
    """
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def get_optimizer_params(
    model, encoder_lr, decoder_lr, weight_decay=0.01, llrd_decay=0.9
):
    """
    Constructs the optimizer parameter groups with Layer-wise Learning Rate Decay (LLRD).

    This function separates parameters into groups based on their layer depth in the backbone
    and applies a decaying learning rate. It also separates parameters for weight decay.

    Args:
        model (torch.nn.Module): The PyTorch model.
        encoder_lr (float): Base learning rate for the encoder (backbone).
        decoder_lr (float): Learning rate for the decoder (head).
        weight_decay (float): Weight decay coefficient.
        llrd_decay (float): Decay factor for LLRD.

    Returns:
        list: List of parameter dictionaries for the optimizer.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Initialize groups dictionary: key=(lr, weight_decay) -> value=list of params
    param_groups = {}

    # Attempt to determine the number of layers in the backbone
    # Default to 24 for DeBERTa-v3-large if config is not accessible
    num_layers = 24
    if hasattr(model, "backbone") and hasattr(model.backbone, "config"):
        num_layers = getattr(model.backbone.config, "num_hidden_layers", 24)
    elif hasattr(model, "config"):
        num_layers = getattr(model.config, "num_hidden_layers", 24)

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Determine Learning Rate
        lr = encoder_lr

        # Check if parameter belongs to the backbone or the head.
        # We assume backbone parameters contain 'backbone', 'deberta', 'roberta', or 'bert'.
        # If none of these are present, we assume it's part of the task-specific head.
        is_backbone = any(k in name for k in ["backbone", "deberta", "roberta", "bert"])

        if not is_backbone:
            lr = decoder_lr
        else:
            # Apply LLRD for backbone
            # Identify layer index from name (e.g., "encoder.layer.11")
            layer_match = re.search(r"layer\.(\d+)", name)
            if layer_match:
                layer_id = int(layer_match.group(1))
                # Calculate decay: higher layers (closer to output) get higher LR
                # Layer (num_layers - 1) gets encoder_lr
                # Layer 0 gets encoder_lr * (decay ** (num_layers - 1))
                lr = encoder_lr * (llrd_decay ** (num_layers - 1 - layer_id))
            elif "embeddings" in name:
                # Embeddings get the lowest LR
                lr = encoder_lr * (llrd_decay**num_layers)
            else:
                # Other backbone parameters (e.g., pooler, final layernorm in backbone)
                # Usually kept at the base encoder_lr
                lr = encoder_lr

        # Determine Weight Decay
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # Group parameters
        group_key = (lr, wd)
        if group_key not in param_groups:
            param_groups[group_key] = []
        param_groups[group_key].append(param)

    # Convert to list of dicts expected by PyTorch optimizers
    optimizer_parameters = []
    for (lr, wd), params in param_groups.items():
        optimizer_parameters.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_parameters

import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the seed for generating random numbers to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_qwk(y_true, y_pred):
    """
    Computes the Quadratic Weighted Kappa.

    Args:
        y_true: Array-like of true labels (integers 1-6).
        y_pred: Array-like of predicted scores (can be floats).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Clip predictions to the valid range [1, 6] and round to nearest integer
    # This converts regression outputs to ordinal classes required for QWK
    y_pred = np.clip(y_pred, 1, 6).round().astype(int)
    y_true = y_true.astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic")


def get_llrd_optimizer_params(model, encoder_lr, head_lr, weight_decay, llrd_decay):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The PyTorch model.
        encoder_lr: Learning rate for the top layer of the encoder.
        head_lr: Learning rate for the task-specific head.
        weight_decay: Weight decay coefficient.
        llrd_decay: Multiplicative decay factor for lower layers.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    no_decay = {"bias", "LayerNorm.bias", "LayerNorm.weight"}

    # 1. Detect the number of layers in the backbone
    # We scan parameter names for 'layer.X' patterns to find the max index.
    max_layer_index = 0
    found_layers = False
    for name, _ in model.named_parameters():
        if "layer." in name:
            parts = name.split(".")
            try:
                # Find the index of 'layer' and get the next element
                idx = parts.index("layer")
                layer_num = int(parts[idx + 1])
                if layer_num > max_layer_index:
                    max_layer_index = layer_num
                found_layers = True
            except (ValueError, IndexError):
                continue

    # If we found layers, the count is max_index + 1. Otherwise assume flat backbone (depth 0).
    num_layers = max_layer_index + 1 if found_layers else 1

    optimizer_grouped_parameters = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # --- 1. Determine Weight Decay ---
        # Do not apply weight decay to bias or LayerNorm parameters
        if any(nd in name for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        # --- 2. Determine Learning Rate ---
        # Default assumption: it's a head parameter
        lr = head_lr

        # Heuristic to identify backbone parameters:
        # Check for model-specific keywords (deberta, roberta, bert) or structural keywords (embeddings, encoder)
        # Note: Config.model_name is "microsoft/deberta-v3-large", so internal name is usually 'deberta'
        is_backbone = False
        if "deberta" in name or "embeddings" in name or "encoder" in name:
            is_backbone = True

        if is_backbone:
            if "embeddings" in name:
                # Embeddings get the lowest LR: encoder_lr * decay^(num_layers)
                lr = encoder_lr * (llrd_decay**num_layers)
            elif "layer." in name:
                # Encoder layers
                try:
                    parts = name.split(".")
                    idx = parts.index("layer")
                    layer_idx = int(parts[idx + 1])

                    # Layer N-1 (top) gets encoder_lr * decay^0
                    # Layer 0 (bottom) gets encoder_lr * decay^(N-1)
                    decay_power = num_layers - 1 - layer_idx
                    lr = encoder_lr * (llrd_decay**decay_power)
                except (ValueError, IndexError):
                    # Fallback if parsing fails
                    lr = encoder_lr
            else:
                # Other backbone parameters (e.g., final LayerNorm after encoder, pooler)
                # Treat them as top-layer parameters
                lr = encoder_lr

        optimizer_grouped_parameters.append(
            {"params": [param], "weight_decay": wd, "lr": lr}
        )

    return optimizer_grouped_parameters

import os
import random
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from library.config import Config


def seed_everything(seed=42):
    """
    Sets the random seed for reproducibility across python, numpy, and torch.
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
    Computes the Quadratic Weighted Kappa (QWK) score.

    Args:
        y_true: Array-like of true scores.
        y_pred: Array-like of predicted scores (can be continuous).

    Returns:
        float: The QWK score.
    """
    # Ensure inputs are numpy arrays
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()

    # Clip and round predictions to match the integer scale 1-6
    y_pred_int = np.rint(np.clip(y_pred, 1, 6)).astype(int)
    y_true_int = np.rint(np.clip(y_true, 1, 6)).astype(int)

    return cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic")


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model: The PyTorch model.
        encoder_lr: Learning rate for the top layer of the backbone.
        decoder_lr: Learning rate for the regression head.
        weight_decay: Weight decay coefficient.

    Returns:
        list: A list of dictionaries containing parameter groups and their specific settings.
    """
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Initialize groups
    optimizer_grouped_parameters = []

    # Detect number of layers dynamically based on parameter names
    # DeBERTa v3 structure usually involves 'encoder.layer.X'
    layer_indices = [
        int(n.split("layer.")[1].split(".")[0])
        for n, p in param_optimizer
        if "layer." in n and "encoder" in n
    ]
    num_layers = max(layer_indices) + 1 if layer_indices else 24

    # Define decay factor from Config
    llrd_decay = Config.llrd_decay

    # Group parameters
    # We group by (lr, weight_decay) to keep the optimizer param list clean,
    # but strictly speaking, a list of dicts per param is also valid.
    # Here we iterate and assign.

    for n, p in param_optimizer:
        if not p.requires_grad:
            continue

        # Determine Learning Rate
        if "embeddings" in n:
            # Embeddings get the strongest decay (furthest from head)
            lr = encoder_lr * (llrd_decay**num_layers)
        elif "encoder.layer." in n:
            # Extract layer index
            try:
                layer_idx = int(n.split("layer.")[1].split(".")[0])
                # Calculate distance from the top.
                # Top layer (index = num_layers - 1) has distance 0 -> lr = encoder_lr
                # Bottom layer (index = 0) has distance num_layers - 1
                distance_from_top = (num_layers - 1) - layer_idx
                lr = encoder_lr * (llrd_decay**distance_from_top)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                lr = encoder_lr
        elif "rel_embeddings" in n:
            # Relative embeddings in DeBERTa
            lr = encoder_lr * (llrd_decay**num_layers)
        else:
            # Head parameters (not in backbone)
            lr = decoder_lr

        # Determine Weight Decay
        if any(nd in n for nd in no_decay):
            wd = 0.0
        else:
            wd = weight_decay

        optimizer_grouped_parameters.append(
            {"params": [p], "lr": lr, "weight_decay": wd}
        )

    return optimizer_grouped_parameters

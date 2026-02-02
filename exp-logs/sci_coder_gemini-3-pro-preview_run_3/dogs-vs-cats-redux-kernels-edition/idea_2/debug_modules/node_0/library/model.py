import torch
import torch.nn as nn
import timm
from library.config import Config
from library.utils import get_logger

logger = get_logger("model")


def get_model(
    model_name=Config.MODEL_NAME,
    num_classes=Config.NUM_CLASSES,
    pretrained=Config.PRETRAINED,
    device=Config.DEVICE,
):
    """
    Creates and returns the ConvNeXt model using timm.

    Args:
        model_name (str): Name of the model in timm.
        num_classes (int): Number of output classes.
        pretrained (bool): Whether to load pretrained weights.
        device (str): Device to put the model on.

    Returns:
        torch.nn.Module: The instantiated model.
    """
    logger.info(f"Creating model: {model_name} (Pretrained={pretrained})")

    try:
        model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=num_classes
        )
    except Exception as e:
        logger.error(f"Failed to create model via timm: {e}")
        raise e

    model.to(device)
    return model


def build_optimizer_params(model, base_lr, weight_decay, decay_rate):
    """
    Constructs parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Logic:
    - Head/Final Norm: lr = base_lr
    - Stage 3: lr = base_lr * decay_rate
    - Stage 2: lr = base_lr * decay_rate^2
    - Stage 1: lr = base_lr * decay_rate^3
    - Stage 0: lr = base_lr * decay_rate^4
    - Stem:    lr = base_lr * decay_rate^5

    Also separates parameters for weight decay (no decay for biases and 1D tensors).
    """

    # Define layer names for grouping
    # ConvNeXt structure in timm: stem, stages.0, stages.1, stages.2, stages.3, norm_pre, head

    param_groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1. Determine Learning Rate Scale based on depth
        scale = 1.0

        if name.startswith("head") or name.startswith("norm"):  # Final layers
            scale = 1.0
        elif name.startswith("stages.3"):
            scale = decay_rate**1
        elif name.startswith("stages.2"):
            scale = decay_rate**2
        elif name.startswith("stages.1"):
            scale = decay_rate**3
        elif name.startswith("stages.0"):
            scale = decay_rate**4
        elif name.startswith("stem"):
            scale = decay_rate**5
        else:
            # Fallback for any other parameters (usually part of the backbone start)
            scale = decay_rate**5

        # 2. Determine Weight Decay
        # Apply weight decay to weights (dim >= 2), skip biases and norms (dim < 2)
        if param.ndim < 2 or "bias" in name or "bn" in name or "norm" in name:
            p_wd = 0.0
        else:
            p_wd = weight_decay

        # Key for grouping: (lr_scale, weight_decay_value)
        key = (scale, p_wd)

        if key not in param_groups:
            param_groups[key] = []

        param_groups[key].append(param)

    # Convert to list of dicts format expected by optimizer
    optimizer_params = []
    for (scale, p_wd), params in param_groups.items():
        optimizer_params.append(
            {"params": params, "lr": base_lr * scale, "weight_decay": p_wd}
        )

    return optimizer_params


def get_optimizer(model):
    """
    Returns the AdamW optimizer configured with LLRD if enabled in Config.
    """
    if Config.USE_LLRD:
        logger.info(
            f"Initializing AdamW with LLRD (Decay Rate: {Config.LLRD_DECAY_RATE})"
        )
        params = build_optimizer_params(
            model,
            base_lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
            decay_rate=Config.LLRD_DECAY_RATE,
        )
    else:
        logger.info("Initializing AdamW with standard parameter grouping")
        # Standard separation of weight decay
        decay = []
        no_decay = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim < 2 or "bias" in name or "bn" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)

        params = [
            {"params": decay, "weight_decay": Config.WEIGHT_DECAY},
            {"params": no_decay, "weight_decay": 0.0},
        ]

    optimizer = torch.optim.AdamW(params, lr=Config.LEARNING_RATE)
    return optimizer


def get_scheduler(optimizer):
    """
    Returns the CosineAnnealingLR scheduler.
    """
    logger.info(f"Initializing CosineAnnealingLR (T_max={Config.SCHEDULER_T_MAX})")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )
    return scheduler


def get_loss_fn():
    """
    Returns CrossEntropyLoss with Label Smoothing.
    """
    logger.info(
        f"Initializing CrossEntropyLoss (Label Smoothing={Config.LABEL_SMOOTHING})"
    )
    return nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

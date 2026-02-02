import torch
import torch.nn as nn
from transformers import (
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from library.config import Config


def get_llrd_optimizer_params(
    model,
    encoder_lr=Config.LR,
    decoder_lr=Config.LR,
    weight_decay=Config.WEIGHT_DECAY,
    llrd_decay=Config.LLRD_DECAY,
):
    """
    Configures the optimizer parameters with Layer-wise Learning Rate Decay (LLRD).
    This assigns different learning rates to different layers of the model,
    typically higher LRs for the head and top layers, and lower LRs for bottom layers.

    Args:
        model: The model to optimize.
        encoder_lr: Learning rate for the top layer of the encoder.
        decoder_lr: Learning rate for the classification head.
        weight_decay: Weight decay coefficient.
        llrd_decay: Decay rate for lower layers (e.g., 0.95).

    Returns:
        List of parameter dictionaries for the optimizer.
    """

    # Identify the backbone config to determine number of layers
    # ToxicityModel wraps the backbone in self.model
    if hasattr(model, "model") and hasattr(model.model, "config"):
        config = model.model.config
    elif hasattr(model, "config"):
        config = model.config
    else:
        # Fallback default
        config = None

    # Determine number of hidden layers
    num_hidden_layers = 12  # Default fallback
    if config:
        if hasattr(config, "num_hidden_layers"):
            num_hidden_layers = config.num_hidden_layers
        elif hasattr(config, "n_layer"):  # GPT style
            num_hidden_layers = config.n_layer

    # Parameter groups
    optimizer_grouped_parameters = []

    # Parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Get all named parameters
    named_parameters = list(model.named_parameters())

    # Track which parameters have been assigned to a group to avoid duplicates
    assigned_ids = set()

    # 1. Classification Head (Decoder)
    # In ToxicityModel, the head is named 'fc'
    head_params_decay = []
    head_params_no_decay = []

    for n, p in named_parameters:
        if "fc" in n:
            if any(nd in n for nd in no_decay):
                head_params_no_decay.append(p)
            else:
                head_params_decay.append(p)
            assigned_ids.add(id(p))

    if head_params_decay:
        optimizer_grouped_parameters.append(
            {
                "params": head_params_decay,
                "weight_decay": weight_decay,
                "lr": decoder_lr,
            }
        )
    if head_params_no_decay:
        optimizer_grouped_parameters.append(
            {
                "params": head_params_no_decay,
                "weight_decay": 0.0,
                "lr": decoder_lr,
            }
        )

    # 2. Encoder Layers (with decay)
    # We iterate from top (num_layers-1) to bottom (0)
    # Top layer gets encoder_lr, bottom gets encoder_lr * (decay ** (num_layers-1))
    for layer_i in range(num_hidden_layers - 1, -1, -1):
        # Calculate LR for this layer
        layer_lr = encoder_lr * (llrd_decay ** (num_hidden_layers - 1 - layer_i))

        layer_params_decay = []
        layer_params_no_decay = []

        # Search string for this layer
        # Matches 'model.encoder.layer.11.' or 'roberta.encoder.layer.11.'
        search_str = f"encoder.layer.{layer_i}."

        for n, p in named_parameters:
            if id(p) in assigned_ids:
                continue

            if search_str in n:
                if any(nd in n for nd in no_decay):
                    layer_params_no_decay.append(p)
                else:
                    layer_params_decay.append(p)
                assigned_ids.add(id(p))

        if layer_params_decay:
            optimizer_grouped_parameters.append(
                {
                    "params": layer_params_decay,
                    "weight_decay": weight_decay,
                    "lr": layer_lr,
                }
            )
        if layer_params_no_decay:
            optimizer_grouped_parameters.append(
                {
                    "params": layer_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": layer_lr,
                }
            )

    # 3. Embeddings and Remaining Parameters
    # Assign the lowest learning rate: encoder_lr * (decay ** num_hidden_layers)
    embed_lr = encoder_lr * (llrd_decay**num_hidden_layers)

    rest_params_decay = []
    rest_params_no_decay = []

    for n, p in named_parameters:
        if id(p) not in assigned_ids:
            if any(nd in n for nd in no_decay):
                rest_params_no_decay.append(p)
            else:
                rest_params_decay.append(p)
            assigned_ids.add(id(p))

    if rest_params_decay:
        optimizer_grouped_parameters.append(
            {
                "params": rest_params_decay,
                "weight_decay": weight_decay,
                "lr": embed_lr,
            }
        )
    if rest_params_no_decay:
        optimizer_grouped_parameters.append(
            {
                "params": rest_params_no_decay,
                "weight_decay": 0.0,
                "lr": embed_lr,
            }
        )

    return optimizer_grouped_parameters


def get_optimizer(
    model,
    learning_rate=Config.LR,
    weight_decay=Config.WEIGHT_DECAY,
    llrd_decay=Config.LLRD_DECAY,
):
    """
    Creates an AdamW optimizer configured with LLRD.

    Args:
        model: The model to optimize.
        learning_rate: Base learning rate.
        weight_decay: Weight decay.
        llrd_decay: Layer-wise decay factor.

    Returns:
        torch.optim.AdamW
    """
    # We use the same base LR for encoder top layer and decoder head
    params = get_llrd_optimizer_params(
        model,
        encoder_lr=learning_rate,
        decoder_lr=learning_rate,
        weight_decay=weight_decay,
        llrd_decay=llrd_decay,
    )

    optimizer = torch.optim.AdamW(params, lr=learning_rate, eps=1e-6)
    return optimizer


def get_scheduler(
    optimizer,
    num_train_steps,
    warmup_ratio=Config.WARMUP_RATIO,
    scheduler_type=Config.SCHEDULER,
):
    """
    Creates a learning rate scheduler.

    Args:
        optimizer: The optimizer.
        num_train_steps: Total number of training steps.
        warmup_ratio: Ratio of steps for warmup.
        scheduler_type: 'cosine' or 'linear'.

    Returns:
        Hugging Face scheduler.
    """
    num_warmup_steps = int(num_train_steps * warmup_ratio)

    if scheduler_type == "cosine":
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    elif scheduler_type == "linear":
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )
    else:
        # Default to cosine
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_train_steps,
        )

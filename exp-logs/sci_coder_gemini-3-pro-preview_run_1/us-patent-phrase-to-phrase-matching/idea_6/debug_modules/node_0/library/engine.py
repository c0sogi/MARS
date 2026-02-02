import torch
import torch.nn as nn
import numpy as np
import time
from scipy.stats import pearsonr
from torch.cuda.amp import autocast, GradScaler
from library.utils import AverageMeter


def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
    """
    Constructs the parameter groups for the optimizer with Layer-wise Learning Rate Decay (LLRD).

    Args:
        model (nn.Module): The model to optimize.
        encoder_lr (float): Base learning rate for the top transformer layer.
        decoder_lr (float): Learning rate for the regression head.
        weight_decay (float): Weight decay coefficient.

    Returns:
        list: List of parameter group dictionaries.
    """
    # Define parameters to exclude from weight decay
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]

    # Get number of layers from model config
    # DeBERTa-v3-Large typically has 24 layers
    num_layers = model.model_config.num_hidden_layers
    decay_rate = model.cfg.llrd_decay

    # Initialize groups
    # Group structure:
    # - Head (Highest LR)
    # - Layers (N-1 down to 0, Decaying LR)
    # - Embeddings (Lowest LR)

    # We will use a dictionary to map layer indices to lists of params
    # keys: 'head', 'embed', 0..N-1
    param_groups = {
        "head": {"params": [], "params_no_decay": []},
        "embed": {"params": [], "params_no_decay": []},
    }
    for i in range(num_layers):
        param_groups[i] = {"params": [], "params_no_decay": []}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_no_decay = any(nd in name for nd in no_decay)

        if "backbone.embeddings" in name:
            key = "embed"
        elif "backbone.encoder.layer" in name:
            # Extract layer index
            # Example: backbone.encoder.layer.11.output.dense.weight
            try:
                parts = name.split(".")
                # Find the index immediately following "layer"
                layer_idx = int(parts[parts.index("layer") + 1])
                key = layer_idx
            except (ValueError, IndexError):
                # Fallback for unexpected naming within encoder (e.g., relative embeddings)
                key = "embed"
        elif "backbone" in name:
            # Other backbone parameters (e.g. final layernorm, pooler, relative attention)
            # Treat as embeddings/low-level features
            key = "embed"
        else:
            # Head parameters (fc, etc.)
            key = "head"

        if is_no_decay:
            param_groups[key]["params_no_decay"].append(param)
        else:
            param_groups[key]["params"].append(param)

    # Create the final list of groups for the optimizer
    optimizer_parameters = []

    # 1. Head
    optimizer_parameters.append(
        {
            "params": param_groups["head"]["params"],
            "weight_decay": weight_decay,
            "lr": decoder_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": param_groups["head"]["params_no_decay"],
            "weight_decay": 0.0,
            "lr": decoder_lr,
        }
    )

    # 2. Layers (Top to Bottom)
    # Layer N-1 gets encoder_lr
    # Layer N-2 gets encoder_lr * decay
    for i in range(num_layers - 1, -1, -1):
        layer_lr = encoder_lr * (decay_rate ** (num_layers - 1 - i))
        optimizer_parameters.append(
            {
                "params": param_groups[i]["params"],
                "weight_decay": weight_decay,
                "lr": layer_lr,
            }
        )
        optimizer_parameters.append(
            {
                "params": param_groups[i]["params_no_decay"],
                "weight_decay": 0.0,
                "lr": layer_lr,
            }
        )

    # 3. Embeddings
    embed_lr = encoder_lr * (decay_rate**num_layers)
    optimizer_parameters.append(
        {
            "params": param_groups["embed"]["params"],
            "weight_decay": weight_decay,
            "lr": embed_lr,
        }
    )
    optimizer_parameters.append(
        {
            "params": param_groups["embed"]["params_no_decay"],
            "weight_decay": 0.0,
            "lr": embed_lr,
        }
    )

    return optimizer_parameters


def train_fn(
    train_loader, model, optimizer, epoch, scheduler, device, cfg, logger=None
):
    """
    Executes one training epoch with Mixed Precision and Gradient Accumulation.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        optimizer: The optimizer.
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        device: Torch device.
        cfg: Configuration object.
        logger: Logger object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    scaler = GradScaler(enabled=cfg.fp16)
    losses = AverageMeter()
    start_time = time.time()

    # Reset gradients
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        batch_size = batch["input_ids"].size(0)

        # Forward pass with Mixed Precision
        with autocast(enabled=cfg.fp16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs["loss"]

            # Scale loss for gradient accumulation
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            # Unscale for gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Record loss (scale back up for logging accuracy)
        loss_val = loss.item() * cfg.gradient_accumulation_steps
        losses.update(loss_val, batch_size)

    elapsed = time.time() - start_time

    log_msg = f"Epoch {epoch+1} - Train Loss: {losses.avg:.6f} - Time: {elapsed:.0f}s"
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return losses.avg


def valid_fn(valid_loader, model, device, cfg, logger=None):
    """
    Executes validation loop and calculates Pearson Correlation.

    Args:
        valid_loader: DataLoader for validation data.
        model: The neural network model.
        device: Torch device.
        cfg: Configuration object.
        logger: Logger object.

    Returns:
        tuple: (average_loss, pearson_score, predictions)
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    labels = []
    start_time = time.time()

    with torch.no_grad():
        for batch in valid_loader:
            for k, v in batch.items():
                batch[k] = v.to(device)

            batch_size = batch["input_ids"].size(0)

            # Forward pass
            with autocast(enabled=cfg.fp16):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch.get("labels"),
                )

            if "loss" in outputs:
                losses.update(outputs["loss"].item(), batch_size)

            # Collect predictions
            logits = outputs["logits"]
            preds.append(logits.view(-1).cpu().numpy())

            if "labels" in batch:
                labels.append(batch["labels"].view(-1).cpu().numpy())

    predictions = np.concatenate(preds)

    pearson_score = 0.0
    if len(labels) > 0:
        targets = np.concatenate(labels)
        # Calculate Pearson Correlation
        pearson_score, _ = pearsonr(targets, predictions)

    elapsed = time.time() - start_time

    log_msg = f"Validation Loss: {losses.avg} - Pearson: {pearson_score} - Time: {elapsed:.0f}s"
    if logger:
        logger.info(log_msg)
    else:
        print(log_msg)

    return losses.avg, pearson_score, predictions

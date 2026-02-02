import time
import gc
import math
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from scipy.stats import pearsonr
from library.utils import AverageMeter, get_logger


def get_optimizer_params(model, cfg):
    """
    Configures layer-wise learning rate decay (LLRD) for the optimizer.
    Assigns different learning rates to different layers of the model.

    Args:
        model: The CustomModel instance.
        cfg: Configuration object.

    Returns:
        list: A list of dictionaries defining parameter groups.
    """
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = []

    # Get the number of layers in the backbone (e.g., 24 for DeBERTa-large)
    num_layers = model.config.num_hidden_layers

    # Initialize groups dictionary: (lr, weight_decay) -> list of params
    groups = {}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Default LR and Weight Decay
        lr = cfg.learning_rate
        weight_decay = cfg.weight_decay

        # 1. Determine Learning Rate based on Layer Depth
        if "model.embeddings" in name:
            # Embeddings get the lowest LR
            lr = cfg.learning_rate * (cfg.llrd_decay**num_layers)

        elif "model.encoder.layer." in name:
            # Encoder layers: higher layers (closer to output) get higher LR
            # Extract layer index from name, e.g., "model.encoder.layer.12.output..."
            try:
                parts = name.split(".")
                # Find the index following "layer"
                layer_idx_pos = parts.index("layer") + 1
                layer_idx = int(parts[layer_idx_pos])

                # Calculate decay exponent: Top layer (23) -> 0, Bottom layer (0) -> 23
                exponent = num_layers - 1 - layer_idx
                lr = cfg.learning_rate * (cfg.llrd_decay**exponent)
            except (ValueError, IndexError):
                # Fallback if parsing fails
                lr = cfg.learning_rate

        elif "fc" in name or "dropouts" in name:
            # Classification Head gets the specific head_lr
            lr = cfg.head_lr

        # 2. Determine Weight Decay
        if any(nd in name for nd in no_decay):
            weight_decay = 0.0

        # 3. Group parameters
        key = (lr, weight_decay)
        if key not in groups:
            groups[key] = []
        groups[key].append(param)

    # Convert groups to list format for optimizer
    for (lr, wd), params in groups.items():
        optimizer_parameters.append({"params": params, "lr": lr, "weight_decay": wd})

    return optimizer_parameters


def train_fn(train_loader, model, optimizer, epoch, scheduler, device, cfg):
    """
    Training loop for one epoch.

    Args:
        train_loader: DataLoader for training data.
        model: The model to train.
        optimizer: The optimizer.
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        device: Device to run on.
        cfg: Configuration object.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    scaler = GradScaler(enabled=True)
    losses = AverageMeter()
    start = time.time()

    # Define Loss Function (MSE)
    loss_fn = nn.MSELoss()

    logger = get_logger(filename=f"{cfg.working_dir}/train_fold.log")

    # Zero gradients initially
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Move batch to device
        for k, v in batch.items():
            batch[k] = v.to(device)

        batch_size = batch["input_ids"].size(0)

        # Forward pass with Mixed Precision
        with autocast(enabled=True):
            # Model output is (batch_size,)
            y_preds = model(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
            )

            # Calculate Loss
            loss = loss_fn(y_preds, batch["labels"])

            # Scale loss for gradient accumulation
            if cfg.gradient_accumulation_steps > 1:
                loss = loss / cfg.gradient_accumulation_steps

        # Backward pass
        scaler.scale(loss).backward()

        # Update weights and scheduler if accumulation steps reached
        if (step + 1) % cfg.gradient_accumulation_steps == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler step
            if scheduler is not None:
                scheduler.step()

            # Reset gradients
            optimizer.zero_grad()

        # Update metrics (multiply by accumulation steps to get actual loss value for logging)
        loss_val = loss.item() * cfg.gradient_accumulation_steps
        losses.update(loss_val, batch_size)

        # Log periodically
        if step % 100 == 0 or step == (len(train_loader) - 1):
            logger.info(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start:.0f}s "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, device, cfg):
    """
    Validation loop.

    Args:
        valid_loader: DataLoader for validation data.
        model: The model to evaluate.
        device: Device to run on.
        cfg: Configuration object.

    Returns:
        tuple: (average_loss, pearson_correlation)
    """
    model.eval()
    loss_fn = nn.MSELoss()
    losses = AverageMeter()

    preds = []
    labels = []

    start = time.time()
    logger = get_logger(filename=f"{cfg.working_dir}/train_fold.log")

    with torch.no_grad():
        for step, batch in enumerate(valid_loader):
            # Move to device
            for k, v in batch.items():
                batch[k] = v.to(device)

            batch_size = batch["input_ids"].size(0)

            # Forward pass
            y_preds = model(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
            )

            loss = loss_fn(y_preds, batch["labels"])
            losses.update(loss.item(), batch_size)

            # Collect predictions and labels for metric calculation
            # Move to CPU and numpy
            preds.append(y_preds.to("cpu").numpy())
            labels.append(batch["labels"].to("cpu").numpy())

    predictions = np.concatenate(preds)
    targets = np.concatenate(labels)

    # Calculate Pearson Correlation
    pearson_score, _ = pearsonr(targets, predictions)

    logger.info(
        f"Validation Result - "
        f"Loss: {losses.avg:.6f} "
        f"Pearson: {pearson_score:.6f} "
        f"Time: {time.time() - start:.0f}s"
    )

    return losses.avg, pearson_score


def inference_fn(test_loader, model, device):
    """
    Inference loop for generating predictions on test data.

    Args:
        test_loader: DataLoader for test data.
        model: The trained model.
        device: Device to run on.

    Returns:
        np.array: Array of predicted scores.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                # 'id' is not a tensor input for the model
                if k != "id":
                    batch[k] = v.to(device)

            y_preds = model(
                batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids")
            )

            # Clip predictions to valid range [0, 1] as scores are bounded
            y_preds = torch.clip(y_preds, 0.0, 1.0)

            preds.append(y_preds.to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions

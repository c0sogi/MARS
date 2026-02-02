import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import get_logger, compute_metrics

# Initialize logger
logger = get_logger(name="engine")


def train_fn(model, data_loader, optimizer, scheduler, device, epoch):
    """
    Executes one training epoch.

    Args:
        model: The PyTorch model to train.
        data_loader: DataLoader for training data.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler.
        device: Device to run training on.
        epoch: Current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()

    # Initialize scaler for Mixed Precision Training
    scaler = torch.cuda.amp.GradScaler(enabled=Config.USE_FP16)

    # Loss function: CrossEntropyLoss works with soft targets (probabilities) in recent PyTorch versions
    # If targets are shape (N, C) and float, it treats them as class probabilities.
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    num_batches = len(data_loader)

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    for batch_idx, data in enumerate(data_loader):
        # Move inputs to device
        input_ids_a = data["input_ids_a"].to(device, non_blocking=True)
        attention_mask_a = data["attention_mask_a"].to(device, non_blocking=True)
        response_mask_a = data["response_mask_a"].to(device, non_blocking=True)

        input_ids_b = data["input_ids_b"].to(device, non_blocking=True)
        attention_mask_b = data["attention_mask_b"].to(device, non_blocking=True)
        response_mask_b = data["response_mask_b"].to(device, non_blocking=True)

        scalars = data["scalars"].to(device, non_blocking=True)
        targets = data["target"].to(device, non_blocking=True)

        # Mixed Precision Forward Pass
        with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
            logits = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                response_mask_a=response_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                response_mask_b=response_mask_b,
                scalars=scalars,
            )

            loss = criterion(logits, targets)

            # Scale loss for gradient accumulation
            if Config.ACCUMULATION_STEPS > 1:
                loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (batch_idx + 1) % Config.ACCUMULATION_STEPS == 0:
            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()

            # Scheduler Step (Update every optimizer step)
            if scheduler is not None:
                scheduler.step()

            # Reset gradients
            optimizer.zero_grad()

        # Track Loss (scale back up if accumulated)
        loss_val = loss.item()
        if Config.ACCUMULATION_STEPS > 1:
            loss_val = loss_val * Config.ACCUMULATION_STEPS

        running_loss += loss_val

        # Log periodically (e.g., every 10% of batches)
        if (batch_idx + 1) % max(1, int(num_batches * 0.1)) == 0:
            avg_loss_so_far = running_loss / (batch_idx + 1)
            logger.info(
                f"Epoch {epoch} | Batch {batch_idx + 1}/{num_batches} | Train Loss: {avg_loss_so_far:.6f}"
            )

    avg_train_loss = running_loss / num_batches
    return avg_train_loss


def eval_fn(model, data_loader, device):
    """
    Evaluates the model on validation or test data.

    Args:
        model: The PyTorch model.
        data_loader: DataLoader for validation/test data.
        device: Device to run evaluation on.

    Returns:
        dict: Dictionary containing loss, metrics, and predictions.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    final_targets = []
    final_preds = []

    with torch.no_grad():
        for data in data_loader:
            # Move inputs to device
            input_ids_a = data["input_ids_a"].to(device, non_blocking=True)
            attention_mask_a = data["attention_mask_a"].to(device, non_blocking=True)
            response_mask_a = data["response_mask_a"].to(device, non_blocking=True)

            input_ids_b = data["input_ids_b"].to(device, non_blocking=True)
            attention_mask_b = data["attention_mask_b"].to(device, non_blocking=True)
            response_mask_b = data["response_mask_b"].to(device, non_blocking=True)

            scalars = data["scalars"].to(device, non_blocking=True)

            # Targets might not exist for test set
            targets = None
            if "target" in data:
                targets = data["target"].to(device, non_blocking=True)

            # Forward Pass (Mixed Precision optional in eval, but saves memory)
            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    response_mask_a=response_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    response_mask_b=response_mask_b,
                    scalars=scalars,
                )

                # Calculate Loss if targets exist
                if targets is not None:
                    loss = criterion(logits, targets)
                    running_loss += loss.item()
                    final_targets.append(targets.cpu().numpy())

            # Convert logits to probabilities
            probs = torch.softmax(logits, dim=1)
            final_preds.append(probs.cpu().numpy())

    # Aggregate results
    final_preds = np.vstack(final_preds)

    result = {"predictions": final_preds}

    if len(final_targets) > 0:
        final_targets = np.vstack(final_targets)
        avg_loss = running_loss / len(data_loader)

        # Compute Metrics
        metrics = compute_metrics(final_targets, final_preds)

        result["loss"] = avg_loss
        result["metrics"] = metrics

        logger.info(f"Validation Loss: {avg_loss}")
        logger.info(f"Validation Metrics: {metrics}")

    return result

import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import AverageMeter, get_logger

# Initialize logger
logger = get_logger("engine")


def train_fn(model, dataloader, optimizer, scheduler, device, scaler, epoch):
    """
    Executes the training loop for a single epoch using Gradient Accumulation and Mixed Precision.

    Args:
        model: The PyTorch model to train.
        dataloader: DataLoader containing training data.
        optimizer: Optimizer for updating model weights.
        scheduler: Learning rate scheduler.
        device: Device to run training on (cuda/cpu).
        scaler: GradScaler for mixed precision training.
        epoch: Current epoch number (for logging).

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    loss_meter = AverageMeter()

    # CrossEntropyLoss supports soft targets (probabilities) directly
    criterion = nn.CrossEntropyLoss()

    num_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
        attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
        token_type_ids_a = batch["token_type_ids_a"].to(device, non_blocking=True)

        input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
        attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
        token_type_ids_b = batch["token_type_ids_b"].to(device, non_blocking=True)

        scalars = batch["scalars"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Mixed Precision Forward Pass
        with torch.amp.autocast(device_type="cuda", enabled=Config.FP16):
            logits = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                token_type_ids_a=token_type_ids_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                token_type_ids_b=token_type_ids_b,
                scalars=scalars,
            )
            loss = criterion(logits, targets)

            # Normalize loss for gradient accumulation
            if Config.GRAD_ACCUM_STEPS > 1:
                loss = loss / Config.GRAD_ACCUM_STEPS

        # Backward pass
        scaler.scale(loss).backward()

        # Optimization Step (perform update every GRAD_ACCUM_STEPS or at end of epoch)
        if (step + 1) % Config.GRAD_ACCUM_STEPS == 0 or (step + 1) == num_steps:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Update weights
            scaler.step(optimizer)
            scaler.update()

            # Clear gradients
            optimizer.zero_grad()

            # Update learning rate
            if scheduler is not None:
                scheduler.step()

        # Update loss meter (multiply back by accum steps to report true batch loss)
        loss_val = (
            loss.item() * Config.GRAD_ACCUM_STEPS
            if Config.GRAD_ACCUM_STEPS > 1
            else loss.item()
        )
        loss_meter.update(loss_val, targets.size(0))

    logger.info(f"Epoch {epoch} - Avg Train Loss: {loss_meter.avg}")
    return loss_meter.avg


def eval_fn(model, dataloader, device):
    """
    Executes the evaluation loop on the validation set.

    Args:
        model: The PyTorch model to evaluate.
        dataloader: DataLoader containing validation data.
        device: Device to run evaluation on.

    Returns:
        tuple: (Average Loss, Predictions as numpy array)
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
            attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
            token_type_ids_a = batch["token_type_ids_a"].to(device, non_blocking=True)

            input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
            attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
            token_type_ids_b = batch["token_type_ids_b"].to(device, non_blocking=True)

            scalars = batch["scalars"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=Config.FP16):
                logits = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    token_type_ids_a=token_type_ids_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    token_type_ids_b=token_type_ids_b,
                    scalars=scalars,
                )
                loss = criterion(logits, targets)

            loss_meter.update(loss.item(), targets.size(0))

            # Convert logits to probabilities
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds.append(probs)

    predictions = np.concatenate(preds)
    logger.info(f"Validation Loss: {loss_meter.avg}")

    return loss_meter.avg, predictions


def inference_fn(model, dataloader, device):
    """
    Generates predictions for a given dataloader (used for Test/TTA).
    Does not compute loss.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader containing test data.
        device: Device to run inference on.

    Returns:
        np.array: Predicted probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
            attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
            token_type_ids_a = batch["token_type_ids_a"].to(device, non_blocking=True)

            input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
            attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
            token_type_ids_b = batch["token_type_ids_b"].to(device, non_blocking=True)

            scalars = batch["scalars"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=Config.FP16):
                logits = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    token_type_ids_a=token_type_ids_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    token_type_ids_b=token_type_ids_b,
                    scalars=scalars,
                )

            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            preds.append(probs)

    return np.concatenate(preds)

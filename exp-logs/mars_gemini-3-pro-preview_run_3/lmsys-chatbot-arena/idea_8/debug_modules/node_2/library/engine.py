import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import AverageMeter, get_logger

# Initialize logger
logger = get_logger("engine")


def train_fn(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Training loop for one epoch.
    """
    model.train()

    # Initialize loss tracker
    loss_meter = AverageMeter()

    # Initialize Scaler for Mixed Precision
    scaler = GradScaler(enabled=Config.USE_FP16)

    # Define Loss Function (CrossEntropyLoss supports soft targets)
    criterion = nn.CrossEntropyLoss()

    # Zero gradients initially
    optimizer.zero_grad()

    num_steps = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move batch to device
        input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
        attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
        input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
        attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
        scalars = batch["scalars"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        # Handle optional token_type_ids
        token_type_ids_a = batch.get("token_type_ids_a")
        if token_type_ids_a is not None:
            token_type_ids_a = token_type_ids_a.to(device, non_blocking=True)

        token_type_ids_b = batch.get("token_type_ids_b")
        if token_type_ids_b is not None:
            token_type_ids_b = token_type_ids_b.to(device, non_blocking=True)

        # Mixed Precision Forward Pass
        with autocast(enabled=Config.USE_FP16):
            logits = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                scalars=scalars,
                token_type_ids_a=token_type_ids_a,
                token_type_ids_b=token_type_ids_b,
            )

            loss = criterion(logits, targets)

            # Normalize loss for gradient accumulation
            if Config.ACCUMULATION_STEPS > 1:
                loss = loss / Config.ACCUMULATION_STEPS

        # Backward Pass with Scaler
        scaler.scale(loss).backward()

        # Update Loss Meter (scale back up for logging)
        loss_val = loss.item()
        if Config.ACCUMULATION_STEPS > 1:
            loss_val = loss_val * Config.ACCUMULATION_STEPS
        loss_meter.update(loss_val, input_ids_a.size(0))

        # Optimizer Step (Accumulated)
        if (step + 1) % Config.ACCUMULATION_STEPS == 0:
            # Unscale gradients for clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

            # Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Logging
        if (step + 1) % 100 == 0 or (step + 1) == num_steps:
            logger.info(
                f"Epoch [{epoch}/{Config.EPOCHS}] "
                f"Step [{step + 1}/{num_steps}] "
                f"Loss: {loss_meter.avg:.6f}"
            )

    return loss_meter.avg


def eval_fn(model, dataloader, device):
    """
    Evaluation loop for validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
            attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
            input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
            attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
            scalars = batch["scalars"].to(device, non_blocking=True)
            targets = batch["target"].to(device, non_blocking=True)

            token_type_ids_a = batch.get("token_type_ids_a")
            if token_type_ids_a is not None:
                token_type_ids_a = token_type_ids_a.to(device, non_blocking=True)

            token_type_ids_b = batch.get("token_type_ids_b")
            if token_type_ids_b is not None:
                token_type_ids_b = token_type_ids_b.to(device, non_blocking=True)

            # Forward Pass (Mixed Precision optional for inference but good for consistency)
            with autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    scalars=scalars,
                    token_type_ids_a=token_type_ids_a,
                    token_type_ids_b=token_type_ids_b,
                )
                loss = criterion(logits, targets)

            loss_meter.update(loss.item(), input_ids_a.size(0))

    logger.info(f"Validation Loss: {loss_meter.avg}")
    return loss_meter.avg


def inference_fn(model, dataloader, device):
    """
    Inference loop for test set. Returns probabilities.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device, non_blocking=True)
            attention_mask_a = batch["attention_mask_a"].to(device, non_blocking=True)
            input_ids_b = batch["input_ids_b"].to(device, non_blocking=True)
            attention_mask_b = batch["attention_mask_b"].to(device, non_blocking=True)
            scalars = batch["scalars"].to(device, non_blocking=True)

            token_type_ids_a = batch.get("token_type_ids_a")
            if token_type_ids_a is not None:
                token_type_ids_a = token_type_ids_a.to(device, non_blocking=True)

            token_type_ids_b = batch.get("token_type_ids_b")
            if token_type_ids_b is not None:
                token_type_ids_b = token_type_ids_b.to(device, non_blocking=True)

            with autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=input_ids_a,
                    attention_mask_a=attention_mask_a,
                    input_ids_b=input_ids_b,
                    attention_mask_b=attention_mask_b,
                    scalars=scalars,
                    token_type_ids_a=token_type_ids_a,
                    token_type_ids_b=token_type_ids_b,
                )

            # Convert logits to probabilities
            probs = torch.softmax(logits.float(), dim=1)
            preds.append(probs.cpu().numpy())

    predictions = np.concatenate(preds, axis=0)
    return predictions

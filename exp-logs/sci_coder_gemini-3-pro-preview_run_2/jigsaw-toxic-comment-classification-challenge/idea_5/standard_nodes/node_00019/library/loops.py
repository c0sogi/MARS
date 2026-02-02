import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.config import Config
from library.utils import AverageMeter, get_score, format_time


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    config=Config,
):
    """
    Executes the training loop for a single epoch.

    Args:
        train_loader: DataLoader for training data.
        model: The neural network model.
        criterion: Loss function (e.g., BCEWithLogitsLoss).
        optimizer: Optimizer (e.g., AdamW).
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        device: Torch device (cuda/cpu).
        awp: Instance of AWP class for adversarial training.
        config: Configuration class.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    start = time.time()

    # Reset gradients at the start
    optimizer.zero_grad()

    num_steps = len(train_loader)

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
        }
        if "token_type_ids" in batch:
            inputs["token_type_ids"] = batch["token_type_ids"].to(device)

        labels = batch["labels"].to(device)
        batch_size = labels.size(0)

        # --- Forward Pass (Clean) ---
        with autocast():
            y_preds = model(**inputs)
            loss = criterion(y_preds, labels)
            # Scale loss for gradient accumulation
            loss = loss / config.GRAD_ACCUM_STEPS

        # --- Backward Pass (Clean) ---
        scaler.scale(loss).backward()

        # --- Adversarial Weight Perturbation (AWP) ---
        if awp is not None and epoch >= config.AWP_START_EPOCH:
            # Attack: Perturb model weights based on accumulated gradients
            awp.attack_step(epoch)

            # Forward Pass (Adversarial)
            with autocast():
                y_preds_adv = model(**inputs)
                loss_adv = criterion(y_preds_adv, labels)
                loss_adv = loss_adv / config.GRAD_ACCUM_STEPS

            # Backward Pass (Adversarial)
            # Accumulate adversarial gradients into existing gradients
            scaler.scale(loss_adv).backward()

            # Restore original weights
            awp.restore()

        # --- Optimizer Step (with Gradient Accumulation) ---
        if (step + 1) % config.GRAD_ACCUM_STEPS == 0:
            # Unscale gradients before clipping (if not already unscaled by AWP)
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            # Step optimizer and scheduler
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

        # Update loss meter (rescale to original magnitude for logging)
        losses.update(loss.item() * config.GRAD_ACCUM_STEPS, batch_size)

        # Logging
        if step % 100 == 0 or step == (num_steps - 1):
            print(
                f"Epoch: [{epoch + 1}][{step}/{num_steps}] "
                f"Elapsed: {format_time(time.time() - start)} "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device, config=Config):
    """
    Executes the validation loop.

    Args:
        val_loader: DataLoader for validation data.
        model: The neural network model.
        criterion: Loss function.
        device: Torch device.
        config: Configuration class.

    Returns:
        tuple: (average_loss, roc_auc_score, predictions_array)
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            labels = batch["labels"].to(device)
            batch_size = labels.size(0)

            y_preds = model(**inputs)
            loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            # Apply sigmoid to convert logits to probabilities
            preds.append(y_preds.sigmoid().to("cpu").numpy())
            targets.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Calculate Metric
    score = get_score(ground_truth, predictions)

    print(f"Validation Loss: {losses.avg:.8f}")
    print(f"Validation ROC AUC: {score:.8f}")
    print(f"Validation Time: {format_time(time.time() - start)}")

    return losses.avg, score, predictions


def inference_fn(test_loader, model, device, config=Config):
    """
    Executes the inference loop on the test set.

    Args:
        test_loader: DataLoader for test data.
        model: The neural network model.
        device: Torch device.
        config: Configuration class.

    Returns:
        np.array: Predicted probabilities for the test set.
    """
    model.eval()
    preds = []
    start = time.time()

    print("Starting Inference...")

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            inputs = {
                "input_ids": batch["input_ids"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            if "token_type_ids" in batch:
                inputs["token_type_ids"] = batch["token_type_ids"].to(device)

            y_preds = model(**inputs)

            # Apply sigmoid to convert logits to probabilities
            preds.append(y_preds.sigmoid().to("cpu").numpy())

            if step % 500 == 0:
                print(f"Inference Step: {step}/{len(test_loader)}")

    predictions = np.concatenate(preds)
    print(f"Inference Complete. Time: {format_time(time.time() - start)}")

    return predictions

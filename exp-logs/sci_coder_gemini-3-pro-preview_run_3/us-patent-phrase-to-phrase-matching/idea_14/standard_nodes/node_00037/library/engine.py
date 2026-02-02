import time
import torch
import numpy as np
from library.utils import AverageMeter, compute_score, get_logger
from library.config import Config

# Initialize logger
logger = get_logger("engine.log")


def train_fn(
    train_loader,
    model,
    optimizer,
    device,
    scheduler,
    epoch,
    config,
    awp=None,
    ema=None,
    loss_fn=None,
    scaler=None,
):
    """
    Executes one training epoch with Mixed Precision and AWP.
    """
    model.train()
    losses = AverageMeter()

    # Ensure gradients are zero before starting
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        batch_size = labels.size(0)

        # -----------------------------------------------------------
        # 1. Standard Forward Pass (Mixed Precision)
        # -----------------------------------------------------------
        with torch.amp.autocast("cuda"):
            outputs = model(input_ids, attention_mask, token_type_ids)
            loss, loss_dict = loss_fn(outputs, labels)

        # -----------------------------------------------------------
        # 2. Standard Backward Pass (Scaled)
        # -----------------------------------------------------------
        scaler.scale(loss).backward()

        # -----------------------------------------------------------
        # 3. Adversarial Weight Perturbation (AWP)
        # -----------------------------------------------------------
        if config.use_awp and awp is not None and awp.should_apply(epoch):
            # AWP works with scaled gradients (direction is preserved)
            awp.attack_step()

            # Forward pass with perturbed weights
            with torch.amp.autocast("cuda"):
                adv_outputs = model(input_ids, attention_mask, token_type_ids)
                adv_loss, _ = loss_fn(adv_outputs, labels)

            # Backward pass with perturbed weights (accumulate scaled gradients)
            scaler.scale(adv_loss).backward()

            # Restore original weights
            awp.restore()

        # -----------------------------------------------------------
        # 4. Optimization Step
        # -----------------------------------------------------------
        # Unscale gradients before clipping
        scaler.unscale_(optimizer)

        # Clip gradients to prevent exploding gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.max_grad_norm
        )

        # Scaler step (skips if NaNs found)
        scaler.step(optimizer)
        scaler.update()

        # Update Learning Rate Scheduler (usually per step for Transformers)
        if scheduler is not None:
            scheduler.step()

        # -----------------------------------------------------------
        # 5. EMA Update
        # -----------------------------------------------------------
        if config.use_ema and ema is not None:
            ema.update()

        # Clear gradients for next step
        optimizer.zero_grad()

        # -----------------------------------------------------------
        # 6. Logging
        # -----------------------------------------------------------
        losses.update(loss.item(), batch_size)

        if step % config.print_freq == 0 or step == (len(train_loader) - 1):
            lr = optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch: [{epoch + 1}][{step}/{len(train_loader)}] "
                f"Loss: {losses.val:.6f} ({losses.avg:.6f}) "
                f"LR: {lr:.8f} "
                f"Grad: {grad_norm:.4f}"
            )

    return losses.avg


def valid_fn(val_loader, model, device, config, ema=None, loss_fn=None):
    """
    Evaluates the model on the validation set.
    Uses EMA weights if available.
    """
    # Apply EMA weights for evaluation
    if config.use_ema and ema is not None:
        logger.info("Applying EMA weights for validation...")
        ema.apply_shadow()

    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            batch_size = labels.size(0)

            # Use autocast for inference memory efficiency
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, token_type_ids)
                loss, _ = loss_fn(outputs, labels)

            losses.update(loss.item(), batch_size)

            # Extract regression logits (first element of tuple)
            logits = outputs[0]

            # Store predictions and targets
            preds.append(logits.view(-1).float().cpu().numpy())
            targets.append(labels.view(-1).float().cpu().numpy())

    # Restore original weights for continued training
    if config.use_ema and ema is not None:
        ema.restore()

    # Concatenate all batches
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # Clip predictions to valid range [0, 1]
    preds = np.clip(preds, 0, 1)

    # Compute Metric
    score = compute_score(targets, preds)

    return losses.avg, score


def inference_fn(test_loader, model, device, config, ema=None):
    """
    Generates predictions for the test set.
    Uses EMA weights if available.
    """
    # Apply EMA weights for inference
    if config.use_ema and ema is not None:
        logger.info("Applying EMA weights for inference...")
        ema.apply_shadow()

    model.eval()
    preds = []

    with torch.no_grad():
        for step, batch in enumerate(test_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(input_ids, attention_mask, token_type_ids)

            # Extract regression logits
            logits = outputs[0]

            preds.append(logits.view(-1).float().cpu().numpy())

    # Restore original weights (good practice, though model might be done)
    if config.use_ema and ema is not None:
        ema.restore()

    preds = np.concatenate(preds)

    # Clip predictions to valid range [0, 1]
    preds = np.clip(preds, 0, 1)

    return preds

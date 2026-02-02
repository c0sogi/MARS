import torch
import numpy as np
import time
from library.config import CFG
from library.utils import AverageMeter, get_score, get_logger

logger = get_logger("engine.log")


def train_fn(
    fold,
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    awp=None,
    ema=None,
):
    """
    Executes one training epoch.
    Handles Two-Stage Warmup, AWP, and EMA updates.
    """
    model.train()

    # Metrics tracking
    losses = AverageMeter()
    mse_losses = AverageMeter()
    ce_losses = AverageMeter()
    pearson_losses = AverageMeter()

    scaler = torch.amp.GradScaler("cuda")

    # --- Two-Stage Warmup Logic ---
    # Stage 1: Freeze backbone (Epoch < warmup_epochs)
    # Stage 2: Unfreeze backbone (Epoch >= warmup_epochs)
    if epoch < CFG.warmup_epochs:
        logger.info(f"Epoch {epoch}: Stage 1 - Freezing Backbone (Head Warmup)")
        for param in model.backbone.parameters():
            param.requires_grad = False
    else:
        # Only log unfreezing once at the transition or start of Stage 2
        if epoch == CFG.warmup_epochs:
            logger.info(
                f"Epoch {epoch}: Stage 2 - Unfreezing Backbone (Full Fine-tuning)"
            )
        for param in model.backbone.parameters():
            param.requires_grad = True

    start = time.time()

    for step, inputs in enumerate(train_loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["label"]
        labels_cls = inputs["label_cls"]
        batch_size = labels.size(0)

        # Forward pass
        with torch.amp.autocast("cuda"):
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs["token_type_ids"],
            )

            # Calculate Loss
            loss, loss_dict = criterion(outputs, labels, labels_cls)

        # Backward pass
        scaler.scale(loss).backward()

        # --- Adversarial Weight Perturbation (AWP) ---
        if awp is not None and epoch >= CFG.awp_start_epoch:
            scaler.unscale_(optimizer)
            # 1. Perturb weights based on gradients
            awp.attack_step()

            # 2. Forward pass with perturbed weights
            with torch.amp.autocast("cuda"):
                adv_outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )

                # 3. Calculate adversarial loss
                adv_loss, _ = criterion(adv_outputs, labels, labels_cls)

            # 4. Backward pass for adversarial loss
            scaler.scale(adv_loss).backward()

            # 5. Restore original weights
            awp.restore()
        else:
            scaler.unscale_(optimizer)

        # Gradient Clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), CFG.max_grad_norm
        )

        # Optimizer Step
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Scheduler Step (Batch-wise)
        if CFG.batch_scheduler:
            scheduler.step()

        # --- Exponential Moving Average (EMA) Update ---
        if ema is not None:
            ema.update()

        # Update metrics
        losses.update(loss_dict["loss"], batch_size)
        mse_losses.update(loss_dict["mse"], batch_size)
        ce_losses.update(loss_dict["ce"], batch_size)
        pearson_losses.update(loss_dict["pearson"], batch_size)

        if step % CFG.print_freq == 0 or step == (len(train_loader) - 1):
            print(
                f"Epoch: [{epoch}][{step}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start:.1f}s "
                f"Loss: {losses.val:.8f}({losses.avg:.8f}) "
                f"MSE: {mse_losses.val:.8f}({mse_losses.avg:.8f}) "
                f"CE: {ce_losses.val:.8f}({ce_losses.avg:.8f}) "
                f"PearsonLoss: {pearson_losses.val:.8f}({pearson_losses.avg:.8f}) "
                f"Grad: {grad_norm:.4f} "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
            )

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, ema=None):
    """
    Evaluates the model on the validation set.
    Uses EMA weights if available.
    """
    # Apply EMA weights for validation
    if ema is not None:
        ema.apply_shadow()

    model.eval()

    losses = AverageMeter()
    preds = []
    targets = []

    start = time.time()

    with torch.no_grad():
        for step, inputs in enumerate(valid_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            labels = inputs["label"]
            labels_cls = inputs["label_cls"]
            batch_size = labels.size(0)

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )

                # Calculate Loss
                loss, loss_dict = criterion(outputs, labels, labels_cls)
            losses.update(loss.item(), batch_size)

            # Collect predictions (Regression output)
            # Clip predictions to [0, 1] range as per task spec
            batch_preds = (
                outputs["logits"].sigmoid().float().cpu().numpy()
                if CFG.target_size == 1 and False
                else outputs["logits"].float().cpu().numpy()
            )
            # Note: The model output is linear. We clip manually later or rely on the linear output matching the 0-1 target range.
            # Since targets are 0-1, the linear layer will learn to output in that range.

            preds.append(batch_preds)
            targets.append(labels.cpu().numpy())

            if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
                print(
                    f"EVAL: [{step}/{len(valid_loader)}] "
                    f"Elapsed: {time.time() - start:.1f}s "
                    f"Loss: {losses.val:.8f}({losses.avg:.8f})"
                )

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0, 1)

    # Calculate Pearson Score
    score = get_score(ground_truth, predictions)

    # Restore original weights for continued training
    if ema is not None:
        ema.restore()

    return losses.avg, score, predictions


def inference_fn(test_loader, model, device, ema=None):
    """
    Generates predictions for the test set.
    Uses EMA weights if available.
    """
    if ema is not None:
        ema.apply_shadow()

    model.eval()

    preds = []

    start = time.time()

    with torch.no_grad():
        for step, inputs in enumerate(test_loader):
            for k, v in inputs.items():
                inputs[k] = v.to(device)

            with torch.amp.autocast("cuda"):
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    token_type_ids=inputs["token_type_ids"],
                )

            batch_preds = outputs["logits"].float().cpu().numpy()
            preds.append(batch_preds)

            if step % CFG.print_freq == 0 or step == (len(test_loader) - 1):
                print(
                    f"TEST: [{step}/{len(test_loader)}] "
                    f"Elapsed: {time.time() - start:.1f}s"
                )

    predictions = np.concatenate(preds)

    # Clip predictions
    predictions = np.clip(predictions, 0, 1)

    if ema is not None:
        ema.restore()

    return predictions

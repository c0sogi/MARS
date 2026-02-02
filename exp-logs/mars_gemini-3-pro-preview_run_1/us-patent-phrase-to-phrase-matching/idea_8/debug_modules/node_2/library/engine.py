import time
import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from library.utils import AverageMeter, get_score, get_logger
from library.config import CFG


class ModelEMA:
    """
    Implements Exponential Moving Average (EMA) for model parameters.
    Maintains a shadow copy of the weights and updates them during training.
    Allows swapping weights for validation/inference to improve generalization.
    """

    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        """Register the initial model parameters as the shadow copy."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update the shadow parameters using the EMA formula."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    1.0 - self.decay
                ) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """Replace model parameters with shadow parameters for inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """Restore original model parameters after inference."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    cfg,
    model_ema=None,
):
    """
    Executes one training epoch.
    """
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    start = time.time()

    # Global step tracking for logging
    global_step = 0

    for step, inputs in enumerate(train_loader):
        # Move inputs to device
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        # Mixed Precision Forward Pass
        with autocast():
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds, labels)

        # Normalize loss for gradient accumulation
        if cfg.accum_iter > 1:
            loss = loss / cfg.accum_iter

        losses.update(loss.item() * cfg.accum_iter, batch_size)

        # Backward Pass
        scaler.scale(loss).backward()

        # Gradient Accumulation Step
        if (step + 1) % cfg.accum_iter == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Update EMA
            if model_ema is not None:
                model_ema.update()

            # Update Scheduler
            if scheduler is not None:
                scheduler.step()

            global_step += 1

        # Logging
        if (step + 1) % cfg.print_freq == 0 or (step + 1) == len(train_loader):
            print(
                f"Epoch: [{epoch + 1}][{step + 1}/{len(train_loader)}] "
                f"Elapsed: {time.time() - start:.1f}s "
                f"Loss: {losses.val:.4f} ({losses.avg:.4f}) "
                f"LR: {scheduler.get_last_lr()[0]:.8f}"
                if scheduler
                else ""
            )

    return losses.avg


def valid_fn(val_loader, model, criterion, device, cfg, model_ema=None):
    """
    Evaluates the model on the validation set.
    Uses EMA weights if provided.
    """
    # Apply EMA weights if available
    if model_ema is not None:
        print("Applying EMA weights for validation...")
        model_ema.apply_shadow()

    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []
    start = time.time()

    for step, inputs in enumerate(val_loader):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        labels = inputs["labels"]
        batch_size = labels.size(0)

        with torch.no_grad():
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )
            loss = criterion(y_preds, labels)

        losses.update(loss.item(), batch_size)

        # Collect predictions and targets for metric calculation
        preds.append(y_preds.to("cpu").numpy())
        targets.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    ground_truth = np.concatenate(targets)

    # Calculate Pearson Correlation
    score = get_score(ground_truth, predictions)

    # Restore original weights if EMA was used
    if model_ema is not None:
        model_ema.restore()

    print(
        f"Validation Result - Epoch Loss: {losses.avg}  Pearson Score: {score}  Time: {time.time() - start:.1f}s"
    )

    return losses.avg, score


def inference_fn(test_loader, model, device, model_ema=None):
    """
    Generates predictions for the test set.
    Uses EMA weights if provided.
    """
    if model_ema is not None:
        print("Applying EMA weights for inference...")
        model_ema.apply_shadow()

    model.eval()
    preds = []

    start = time.time()

    for step, inputs in enumerate(test_loader):
        for k, v in inputs.items():
            if k != "ids":  # Skip non-tensor data
                inputs[k] = v.to(device)

        with torch.no_grad():
            y_preds = model(
                inputs["input_ids"],
                inputs["attention_mask"],
                inputs.get("token_type_ids"),
            )

        preds.append(y_preds.to("cpu").numpy())

    predictions = np.concatenate(preds)

    if model_ema is not None:
        model_ema.restore()

    print(f"Inference complete. Time: {time.time() - start:.1f}s")

    return predictions

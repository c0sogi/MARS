import time
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from library.utils import get_logger

logger = get_logger("engine")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(dataloader, model, optimizer, device, scheduler, epoch, config):
    """
    Standard training loop for Stage 1 (Teacher Training).
    Does not use AWP.
    """
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    start_time = time.time()

    criterion = nn.BCEWithLogitsLoss()

    # Zero gradients at start of epoch
    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)
        batch_size = input_ids.size(0)

        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), targets.view(-1))

            # Normalize loss for gradient accumulation
            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # Update metrics (scale loss back up for logging)
        losses.update(loss.item() * config.gradient_accumulation_steps, batch_size)

        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch+1} - avg_train_loss: {losses.avg:.10f}  time: {elapsed:.0f}s"
    )

    return losses.avg


def train_fn_awp(dataloader, model, optimizer, device, scheduler, epoch, config, awp):
    """
    Adversarial training loop for Stage 2 (Student Distillation).
    Uses AWP to minimize worst-case loss.
    """
    model.train()
    scaler = GradScaler()
    losses = AverageMeter()
    start_time = time.time()

    criterion = nn.BCEWithLogitsLoss()

    optimizer.zero_grad()

    for step, data in enumerate(dataloader):
        input_ids = data["input_ids"].to(device)
        attention_mask = data["attention_mask"].to(device)
        targets = data["target"].to(device)
        batch_size = input_ids.size(0)

        # --- 1. Clean Pass ---
        with autocast():
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs.view(-1), targets.view(-1))

            if config.gradient_accumulation_steps > 1:
                loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        # --- 2. Adversarial Pass ---
        # Only execute if AWP is enabled and start epoch reached
        if config.use_awp and epoch >= config.awp_start_epoch:
            # Save weights and perturb based on current gradients
            awp.attack()

            with autocast():
                # Forward pass with perturbed weights
                adv_outputs = model(input_ids, attention_mask)
                adv_loss = criterion(adv_outputs.view(-1), targets.view(-1))

                if config.gradient_accumulation_steps > 1:
                    adv_loss = adv_loss / config.gradient_accumulation_steps

            # Accumulate adversarial gradients
            # We don't zero_grad here, so we optimize for Clean + Adversarial Loss
            scaler.scale(adv_loss).backward()

            # Restore original weights
            awp.restore()

        # Update metrics (logging the clean loss)
        losses.update(loss.item() * config.gradient_accumulation_steps, batch_size)

        # --- 3. Optimizer Step ---
        if (step + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if scheduler is not None:
                scheduler.step()

    elapsed = time.time() - start_time
    logger.info(
        f"Epoch {epoch+1} - avg_train_loss: {losses.avg:.10f}  time: {elapsed:.0f}s"
    )

    return losses.avg


def eval_fn(dataloader, model, device):
    """
    Evaluation loop for validation and inference.
    Returns average loss and predictions.
    """
    model.eval()
    losses = AverageMeter()
    preds = []

    criterion = nn.BCEWithLogitsLoss()

    start_time = time.time()

    with torch.no_grad():
        for step, data in enumerate(dataloader):
            input_ids = data["input_ids"].to(device)
            attention_mask = data["attention_mask"].to(device)

            # Targets might not be present in test set
            targets = None
            if "target" in data:
                targets = data["target"].to(device)

            with autocast():
                outputs = model(input_ids, attention_mask)

                if targets is not None:
                    loss = criterion(outputs.view(-1), targets.view(-1))
                    losses.update(loss.item(), input_ids.size(0))

            # Apply sigmoid to logits to get probabilities
            batch_preds = torch.sigmoid(outputs.view(-1)).detach().cpu().numpy()
            preds.append(batch_preds)

    predictions = np.concatenate(preds)

    elapsed = time.time() - start_time
    # Only log if we actually computed loss (validation)
    if losses.count > 0:
        logger.info(f"EVAL: avg_val_loss: {losses.avg:.10f}  time: {elapsed:.0f}s")

    return losses.avg, predictions

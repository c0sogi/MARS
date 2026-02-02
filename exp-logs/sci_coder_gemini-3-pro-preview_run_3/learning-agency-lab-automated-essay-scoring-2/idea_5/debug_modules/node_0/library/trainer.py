import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from library.config import Config
from library.utils import compute_qwk
from library.awp import AWP
from library.model import EssayModel, get_optimizer_params


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


def train_fn(
    train_loader,
    model,
    criterion,
    optimizer,
    epoch,
    scheduler,
    device,
    config,
    awp,
    scaler,
):
    """
    Performs one epoch of training.
    """
    model.train()
    losses = AverageMeter()

    for step, batch in enumerate(train_loader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        batch_size = labels.size(0)

        # Mixed Precision Forward Pass
        with torch.amp.autocast("cuda", enabled=config.use_amp):
            y_preds = model(input_ids, attention_mask)
            loss = criterion(y_preds, labels)

        if config.gradient_accumulation_steps > 1:
            loss = loss / config.gradient_accumulation_steps

        # Backward Pass (Scale Loss)
        scaler.scale(loss).backward()

        if (step + 1) % config.gradient_accumulation_steps == 0:
            # Adversarial Weight Perturbation (AWP)
            if config.use_awp:
                awp.attack_backward(input_ids, attention_mask, labels, criterion, epoch)

            # Unscale Gradients
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step
            if scheduler is not None:
                scheduler.step()

        losses.update(loss.item() * config.gradient_accumulation_steps, batch_size)

    return losses.avg


def valid_fn(valid_loader, model, criterion, device, config):
    """
    Performs validation on the validation set.
    """
    model.eval()
    losses = AverageMeter()
    preds = []
    labels_list = []

    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            batch_size = labels.size(0)

            with torch.amp.autocast("cuda", enabled=config.use_amp):
                y_preds = model(input_ids, attention_mask)
                loss = criterion(y_preds, labels)

            losses.update(loss.item(), batch_size)

            # Collect predictions and labels
            preds.append(y_preds.to("cpu").numpy())
            labels_list.append(labels.to("cpu").numpy())

    predictions = np.concatenate(preds)
    targets = np.concatenate(labels_list)

    return losses.avg, predictions, targets


def inference_fn(test_loader, model, device, config):
    """
    Generates predictions for the test set.
    """
    model.eval()
    preds = []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            with torch.amp.autocast("cuda", enabled=config.use_amp):
                y_preds = model(input_ids, attention_mask)

            preds.append(y_preds.to("cpu").numpy())

    predictions = np.concatenate(preds)
    return predictions


def run_fold(fold, train_loader, valid_loader, config, logger):
    """
    Orchestrates the training process for a single fold.
    """
    logger.info(f"-------- Starting Training for Fold {fold} --------")

    device = config.device

    # Initialize Model
    model = EssayModel(config)
    model.to(device)

    # Optimizer with Layer-wise Learning Rate Decay
    optimizer_parameters = get_optimizer_params(model, config)
    optimizer = AdamW(optimizer_parameters, lr=config.learning_rate, eps=1e-6)

    # Scheduler
    num_train_steps = int(
        len(train_loader) * config.epochs / config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_train_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Loss Function (MSE for regression)
    criterion = nn.MSELoss()

    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler("cuda", enabled=config.use_amp)

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=config.awp_lr,
        adv_eps=config.awp_eps,
        start_epoch=config.awp_start_epoch,
        scaler=scaler,
    )

    best_loss = float("inf")
    best_qwk = -1.0
    best_preds = None

    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        avg_train_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            scheduler,
            device,
            config,
            awp,
            scaler,
        )

        # Validate
        avg_val_loss, preds, targets = valid_fn(
            valid_loader, model, criterion, device, config
        )

        # Metrics
        val_qwk = compute_qwk(targets, preds)
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val QWK: {val_qwk:.4f}"
        )

        # Save Best Model (Monitoring Validation Loss)
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_qwk = val_qwk
            best_preds = preds

            save_path = os.path.join(config.checkpoint_dir, f"backbone_fold_{fold}.pth")
            torch.save(model.state_dict(), save_path)
            logger.info(f"Found better model. Saved to {save_path}")

    logger.info(
        f"Fold {fold} Best Val Loss: {best_loss:.4f} | Best Val QWK: {best_qwk:.4f}"
    )

    return best_preds

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from library.utils import AverageMeter, get_device


class LabelSmoothingLoss(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    Handles both hard targets (indices) and soft targets (probabilities) from MixUp/CutMix.
    """

    def __init__(self, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing

    def forward(self, preds, targets):
        """
        Args:
            preds: (batch_size, num_classes) logits
            targets: (batch_size) indices OR (batch_size, num_classes) soft probabilities
        """
        n_classes = preds.size(1)
        log_preds = F.log_softmax(preds, dim=1)

        if targets.dim() == 1:
            # Hard labels: Create smoothed one-hot targets
            with torch.no_grad():
                true_dist = torch.zeros_like(preds)
                true_dist.fill_(self.smoothing / n_classes)
                true_dist.scatter_(
                    1,
                    targets.data.unsqueeze(1),
                    1.0 - self.smoothing + (self.smoothing / n_classes),
                )

            # KL Divergence equivalent: -sum(target * log_pred)
            loss = torch.sum(-true_dist * log_preds, dim=1).mean()
        else:
            # Soft labels (already mixed/smoothed): Use as is
            loss = torch.sum(-targets * log_preds, dim=1).mean()

        return loss


def train_one_epoch(
    epoch, model, train_loader, criterion, optimizer, device, scaler, logger, config
):
    """
    Trains the model for one epoch using Automatic Mixed Precision (AMP).
    """
    model.train()
    loss_meter = AverageMeter()

    start_time = time.time()

    for step, (imgs, targets) in enumerate(train_loader):
        imgs = imgs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        with autocast(enabled=config.AMP):
            logits = model(imgs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()

        if config.MAX_GRAD_NORM > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        scaler.step(optimizer)
        scaler.update()

        loss_meter.update(loss.item(), imgs.size(0))

    elapsed = time.time() - start_time
    logger.info(f"Epoch {epoch} [Train] Loss: {loss_meter.avg} Time: {elapsed}")

    return loss_meter.avg


def validate_one_epoch(epoch, model, val_loader, criterion, device, logger):
    """
    Validates the model on the validation set.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    start_time = time.time()

    with torch.no_grad():
        for imgs, targets in val_loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            logits = model(imgs)
            loss = criterion(logits, targets)

            preds = torch.argmax(logits, dim=1)
            acc = (preds == targets).float().mean()

            loss_meter.update(loss.item(), imgs.size(0))
            acc_meter.update(acc.item(), imgs.size(0))

    elapsed = time.time() - start_time
    # Printing full precision as requested
    logger.info(
        f"Epoch {epoch} [Valid] Loss: {loss_meter.avg} Accuracy: {acc_meter.avg} Time: {elapsed}"
    )

    return loss_meter.avg, acc_meter.avg


class Trainer:
    """
    Orchestrates the training process, including optimization, logging, and early stopping.
    """

    def __init__(self, model, train_loader, val_loader, config, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.logger = logger
        self.device = get_device()

        self.criterion = LabelSmoothingLoss(smoothing=config.LABEL_SMOOTHING)
        self.scaler = GradScaler(enabled=config.AMP)

        self.model.to(self.device)

    def fit(self, optimizer, scheduler, epochs, save_name="best_model.pth"):
        """
        Runs the training loop for the specified number of epochs.
        """
        best_acc = 0.0
        patience_counter = 0
        save_path = os.path.join(self.config.WORKING_DIR, save_name)

        for epoch in range(1, epochs + 1):
            # Training Phase
            train_loss = train_one_epoch(
                epoch,
                self.model,
                self.train_loader,
                self.criterion,
                optimizer,
                self.device,
                self.scaler,
                self.logger,
                self.config,
            )

            # Validation Phase
            val_loss, val_acc = validate_one_epoch(
                epoch,
                self.model,
                self.val_loader,
                self.criterion,
                self.device,
                self.logger,
            )

            # Scheduler Step
            if scheduler is not None:
                scheduler.step()

            # Checkpoint & Early Stopping Logic
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                self.logger.info(
                    f"Saved best model to {save_path} with Accuracy: {best_acc}"
                )
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{self.config.PATIENCE}"
                )

            if patience_counter >= self.config.PATIENCE:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        return best_acc

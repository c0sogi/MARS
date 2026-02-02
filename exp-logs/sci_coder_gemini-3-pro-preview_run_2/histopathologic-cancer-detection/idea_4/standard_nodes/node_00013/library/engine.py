import os
import copy
import torch
import torch.nn as nn
import numpy as np
from library.config import Config
from library.utils import MetricMonitor, calculate_auc
from library.dataset import mixup_data, mixup_criterion


class ModelEma:
    """
    Maintains a moving average of model parameters using an exponential decay.
    """

    def __init__(self, model, decay=0.9999, device=None):
        self.module = copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
        self.module.to(self.device)

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(
                self.module.state_dict().values(), model.state_dict().values()
            ):
                if self.device is not None:
                    model_v = model_v.to(device=self.device)
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        self._update(
            model, update_fn=lambda e, m: self.decay * e + (1.0 - self.decay) * m
        )

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


def train_one_epoch(
    model, train_loader, optimizer, device, scheduler=None, ema_model=None
):
    """
    Trains the model for one epoch using Mixup.
    """
    model.train()
    metric_monitor = MetricMonitor()
    criterion = nn.BCEWithLogitsLoss()

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Apply Mixup
        mixed_images, labels_a, labels_b, lam = mixup_data(
            images, labels, alpha=Config.mixup_alpha
        )

        optimizer.zero_grad()

        # Forward pass
        logits = model(mixed_images)

        # Calculate Loss
        loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Update EMA
        if ema_model is not None:
            ema_model.update(model)

        # Step Scheduler (if it's a batch-level scheduler like OneCycleLR,
        # though Config suggests CosineAnnealingLR which is usually epoch-level.
        # We'll leave this flexible, but typically Cosine is stepped per epoch in main loop)
        # However, some implementations step per iteration. Assuming epoch-level based on Config.

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.metrics["Loss"]["avg"]


def validate(model, val_loader, device, criterion):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    metric_monitor = MetricMonitor()

    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            # Sigmoid for probability
            probs = torch.sigmoid(logits)

            metric_monitor.update("Loss", loss.item())

            preds.extend(probs.cpu().numpy().flatten().tolist())
            targets.extend(labels.cpu().numpy().flatten().tolist())

    auc = calculate_auc(targets, preds)
    return metric_monitor.metrics["Loss"]["avg"], auc


def train_loop(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    fold,
    epochs=Config.epochs,
    patience=5,
):
    """
    Orchestrates the training process with Early Stopping and Checkpointing.
    """
    best_auc = 0.0
    patience_counter = 0

    # Initialize EMA
    ema_model = None
    if Config.use_ema:
        ema_model = ModelEma(model, decay=Config.ema_decay, device=device)

    criterion = nn.BCEWithLogitsLoss()

    # Ensure checkpoint directory exists
    os.makedirs(Config.checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(Config.checkpoint_dir, f"best_model_fold_{fold}.pth")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device, scheduler, ema_model
        )

        # Step scheduler (Epoch-level)
        if scheduler is not None:
            scheduler.step()

        # Validate (Use EMA model if available)
        val_model = ema_model.module if ema_model else model
        val_loss, val_auc = validate(val_model, val_loader, device, criterion)

        # Print Metrics (Full Precision)
        print(
            f"Fold: {fold} | Epoch: {epoch} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val AUC: {val_auc}"
        )

        # Checkpointing and Early Stopping
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(val_model.state_dict(), best_model_path)
            # print(f"New best model saved with AUC: {best_auc}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    return best_auc

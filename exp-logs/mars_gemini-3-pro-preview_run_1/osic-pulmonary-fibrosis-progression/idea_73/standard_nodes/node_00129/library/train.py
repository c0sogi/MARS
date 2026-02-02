import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import Config
from library.data import get_dataloaders
from library.model import PCCGNet
from library.utils import (
    seed_everything,
    AverageMeter,
    LaplaceLogLikelihoodLoss,
    calculate_metric,
)


class EarlyStopping:
    """
    Early stopping to stop the training when the score does not improve after
    certain epochs.
    """

    def __init__(
        self,
        patience=7,
        verbose=False,
        delta=0,
        path="checkpoint.pth",
        trace_func=print,
    ):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
            trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_score_max = -np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_score, model):
        score = val_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_score, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(
                    f"EarlyStopping counter: {self.counter} out of {self.patience}"
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_score, model)
            self.counter = 0

    def save_checkpoint(self, val_score, model):
        """Saves model when validation score increase."""
        if self.verbose:
            self.trace_func(
                f"Validation score increased ({self.val_score_max:.6f} --> {val_score:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_score_max = val_score


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Handles the training of one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move inputs to device
        axial = inputs["axial"].to(device)
        coronal = inputs["coronal"].to(device)
        tabular = inputs["tabular"].to(device)
        delta_week = inputs["delta_week"].to(device)
        base_fvc = inputs["base_fvc"].to(device)
        targets = targets.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(axial, coronal, tabular, delta_week, base_fvc)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if Config.CLIP_GRAD_NORM > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        optimizer.step()

        # Update stats
        losses.update(loss.item(), axial.size(0))

    return losses.avg


def validate_one_epoch(model, loader, criterion, device):
    """
    Handles the validation of one epoch.
    """
    model.eval()
    losses = AverageMeter()
    metrics = AverageMeter()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            # Move inputs to device
            axial = inputs["axial"].to(device)
            coronal = inputs["coronal"].to(device)
            tabular = inputs["tabular"].to(device)
            delta_week = inputs["delta_week"].to(device)
            base_fvc = inputs["base_fvc"].to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(axial, coronal, tabular, delta_week, base_fvc)

            # Compute loss
            loss = criterion(preds, targets)

            # Compute metric
            score = calculate_metric(preds, targets)

            # Update stats
            losses.update(loss.item(), axial.size(0))
            metrics.update(score, axial.size(0))

    return losses.avg, metrics.avg


def run_training():
    """
    Main entry point for training the model.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Starting training on device: {device}")

    # Ensure checkpoint directory exists
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)

    # 2. Data
    print("Loading dataloaders...")
    train_loader, val_loader, _ = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    # 3. Model
    print("Initializing PCCG-Net...")
    model = PCCGNet()
    model = model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Early Stopping
    # We want to maximize the metric (which is negative, closer to 0 is better)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    early_stopping = EarlyStopping(
        patience=Config.PATIENCE, verbose=True, path=checkpoint_path
    )

    # 6. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = validate_one_epoch(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Stats
        # Printing full precision for validation metric as requested
        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Metric: {val_metric}"
        )

        # Check Early Stopping
        early_stopping(val_metric, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best model saved to {checkpoint_path}")

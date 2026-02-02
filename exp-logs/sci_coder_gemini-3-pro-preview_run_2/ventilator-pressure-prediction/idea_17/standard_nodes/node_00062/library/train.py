import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import MetricMonitor, save_checkpoint
from library.dataset import get_data_loaders
from library.model import PCGIBiLSTM


class WeightedL1Loss(nn.Module):
    """
    Weighted L1 Loss function.
    Assigns different weights to inspiratory (u_out=0) and expiratory (u_out=1) phases.
    """

    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss(reduction="none")
        self.w_insp = Config.LOSS_INSPIRATORY_WEIGHT
        self.w_exp = Config.LOSS_EXPIRATORY_WEIGHT

    def forward(self, preds, targets, u_out):
        """
        Args:
            preds: Model predictions (Batch, Seq)
            targets: Ground truth pressure (Batch, Seq)
            u_out: Expiratory valve status (Batch, Seq), 0=Inspiratory, 1=Expiratory
        """
        loss = self.criterion(preds, targets)

        # Calculate weights:
        # If u_out == 0 (Inspiratory) -> weight = w_insp
        # If u_out == 1 (Expiratory) -> weight = w_exp
        weights = (1 - u_out) * self.w_insp + u_out * self.w_exp

        weighted_loss = (loss * weights).mean()
        return weighted_loss


def train_epoch(model, loader, optimizer, criterion, device, epoch):
    """
    Runs one training epoch.
    """
    model.train()
    metric_monitor = MetricMonitor()

    for batch_idx, (X, u_out, y) in enumerate(loader):
        X = X.to(device, non_blocking=True)
        u_out = u_out.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()

        preds = model(X)
        loss = criterion(preds, y, u_out)

        loss.backward()
        optimizer.step()

        metric_monitor.update("Loss", loss.item())

    return metric_monitor.avg_metrics


def validate_epoch(model, loader, criterion, device):
    """
    Runs one validation epoch.
    Calculates Loss (Weighted L1) and MAE (Inspiratory Phase only - competition metric).
    """
    model.eval()
    metric_monitor = MetricMonitor()

    with torch.no_grad():
        for batch_idx, (X, u_out, y) in enumerate(loader):
            X = X.to(device, non_blocking=True)
            u_out = u_out.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            preds = model(X)

            # 1. Weighted Loss (Optimization Objective)
            loss = criterion(preds, y, u_out)
            metric_monitor.update("Loss", loss.item())

            # 2. Inspiratory MAE (Competition Metric)
            # Filter for u_out == 0
            insp_mask = u_out == 0
            if insp_mask.sum() > 0:
                abs_diff = torch.abs(preds - y)
                insp_mae = abs_diff[insp_mask].mean()
                metric_monitor.update("MAE", insp_mae.item())

    return metric_monitor.avg_metrics


def run_training():
    """
    Main training routine.
    Initializes model, optimizer, scheduler, and runs the training loop with early stopping.
    """
    print(f"Initializing training on device: {Config.DEVICE}")

    # 1. Data Loaders
    train_loader, val_loader = get_data_loaders(load_cached_data=True)

    # 2. Model
    model = PCGIBiLSTM()
    model = model.to(Config.DEVICE)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Stretched Horizon: T_max matches total epochs
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.SCHEDULER_T_MAX, eta_min=Config.SCHEDULER_ETA_MIN
    )

    # 4. Loss Function
    criterion = WeightedL1Loss().to(Config.DEVICE)

    # 5. Training Loop
    best_mae = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, Config.DEVICE, epoch
        )

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, Config.DEVICE)

        # Step Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Timing
        duration = time.time() - start_time

        # Logging
        # Printing full precision as requested
        print(f"Epoch {epoch} | Time: {duration:.2f}s | LR: {current_lr:.2e}")
        print(f"  Train Loss: {train_metrics['Loss']}")
        print(f"  Val Loss:   {val_metrics['Loss']}")
        print(f"  Val MAE:    {val_metrics['MAE']}")

        # Checkpointing & Early Stopping
        current_mae = val_metrics["MAE"]

        if current_mae < best_mae:
            best_mae = current_mae
            patience_counter = 0
            print(f"  New Best MAE! Saving model...")
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "best_mae": best_mae,
                },
                is_best=True,
            )
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MAE: {best_mae}")

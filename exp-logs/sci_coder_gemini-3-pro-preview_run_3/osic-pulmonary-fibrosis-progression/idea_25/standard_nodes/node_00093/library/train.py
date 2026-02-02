import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.data import get_train_val_datasets
from library.model import DSPRNet
from library.utils import seed_everything, inverse_transform, laplace_log_likelihood


class MetricAlignedLoss(nn.Module):
    """
    Implements the Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |y_true - y_pred|) / sigma + ln(sqrt(2) * sigma)

    This loss is applied in the scaled (Z-score) space during training.
    """

    def __init__(self):
        super(MetricAlignedLoss, self).__init__()
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, pred_mu, pred_sigma, target):
        """
        Args:
            pred_mu: Predicted FVC (scaled).
            pred_sigma: Predicted Uncertainty (scaled).
            target: Ground truth FVC (scaled).
        """
        # Ensure sqrt_2 is on the correct device
        sqrt_2 = self.sqrt_2.to(pred_mu.device)

        # Calculate absolute error
        delta = torch.abs(target - pred_mu)

        # Calculate loss terms
        # Term 1: (sqrt(2) * delta) / sigma
        term1 = (sqrt_2 * delta) / pred_sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(sqrt_2 * pred_sigma)

        # Sum and mean
        loss = term1 + term2
        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (images, tabular, targets) in enumerate(loader):
        images = images.to(device)
        tabular = tabular.to(device)
        targets = targets.to(device)

        # Flatten targets if necessary (Batch, 1) -> (Batch)
        targets = targets.view(-1)

        optimizer.zero_grad()

        # Forward pass
        mu, sigma = model(images, tabular)

        # Compute loss
        loss = criterion(mu, sigma, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Validation loop. Computes Loss (scaled) and Metric (unscaled/ml).
    """
    model.eval()
    running_loss = 0.0

    # Lists to store unscaled predictions for metric calculation
    all_mu_ml = []
    all_sigma_ml = []
    all_targets_ml = []

    with torch.no_grad():
        for images, tabular, targets in loader:
            images = images.to(device)
            tabular = tabular.to(device)
            targets = targets.to(device)
            targets_flat = targets.view(-1)

            # Forward pass
            mu, sigma = model(images, tabular)

            # Compute Scaled Loss
            loss = criterion(mu, sigma, targets_flat)
            running_loss += loss.item() * images.size(0)

            # Inverse Transform for Metric Calculation
            mu_ml, sigma_ml = inverse_transform(mu.cpu().numpy(), sigma.cpu().numpy())

            # Inverse transform targets: target_ml = target_scaled * std + mean
            targets_np = targets_flat.cpu().numpy()
            targets_ml = targets_np * Config.TARGET_STD + Config.TARGET_MEAN

            all_mu_ml.extend(mu_ml)
            all_sigma_ml.extend(sigma_ml)
            all_targets_ml.extend(targets_ml)

    epoch_loss = running_loss / len(loader.dataset)

    # Compute Competition Metric
    metric_score = laplace_log_likelihood(
        y_true=np.array(all_targets_ml),
        y_pred=np.array(all_mu_ml),
        sigma=np.array(all_sigma_ml),
    )

    return epoch_loss, metric_score


def train_model(debug=Config.DEBUG):
    """
    Main training pipeline.
    """
    seed_everything(Config.SEED)

    device = torch.device(Config.DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    train_ds, val_ds = get_train_val_datasets()

    if debug:
        # Subset for debugging
        indices = list(range(32))
        train_ds = torch.utils.data.Subset(train_ds, indices)
        val_ds = torch.utils.data.Subset(val_ds, indices)
        print("Debug mode: Using subset of data.")

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    model = DSPRNet().to(device)

    # 3. Optimizer with Differential Learning Rates
    # Group parameters
    backbone_params = list(model.visual_branch.parameters())
    # Collect rest of the parameters
    head_params = (
        list(model.tab_encoder.parameters())
        + list(model.deep_mlp.parameters())
        + list(model.linear_stream.parameters())
        + list(model.head.parameters())
    )

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Loss Function
    criterion = MetricAlignedLoss()

    # 6. Training Loop
    best_metric = -float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_metric = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        current_lr_backbone = optimizer.param_groups[0]["lr"]
        current_lr_head = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val Metric: {val_metric:.5f} | "
            f"LR: {current_lr_backbone:.2e}/{current_lr_head:.2e}"
        )

        # Checkpointing (Maximize Metric - higher is better, negative values)
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New Best Metric! Model saved to {save_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Validation Metric: {best_metric:.5f}")

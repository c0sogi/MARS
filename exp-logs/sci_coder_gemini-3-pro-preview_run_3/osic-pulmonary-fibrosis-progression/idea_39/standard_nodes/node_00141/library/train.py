import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.utils import seed_everything, metric_score
from library.data import get_data
from library.model import CI_OP_DS_Net


class SmoothLaplaceLoss(nn.Module):
    """
    Implements the Smooth Metric-Aligned Laplace Log Likelihood Loss.
    Formula: L = (sqrt(2) * |true - pred|) / sigma + ln(sqrt(2) * sigma)

    As per the idea description:
    - Explicitly includes sqrt(2) constants.
    - Removes all clipping/thresholding (min, clamp) to prevent gradient vanishing on outliers.
    """

    def __init__(self):
        super(SmoothLaplaceLoss, self).__init__()
        # Register sqrt(2) as a buffer so it moves with the model/loss to device
        self.register_buffer("sqrt_2", torch.sqrt(torch.tensor(2.0)))

    def forward(self, pred_mu, pred_sigma, target):
        # pred_mu: (B,)
        # pred_sigma: (B,) - Guaranteed positive by model (Softplus + epsilon)
        # target: (B,)

        delta = torch.abs(target - pred_mu)

        # Calculate NLL terms
        # Term 1: sqrt(2) * delta / sigma
        term1 = (self.sqrt_2 * delta) / pred_sigma

        # Term 2: ln(sqrt(2) * sigma)
        term2 = torch.log(self.sqrt_2 * pred_sigma)

        # Mean over batch
        loss = torch.mean(term1 + term2)

        return loss


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (img, meta_a, meta_b, target) in enumerate(loader):
        img = img.to(device)
        meta_a = meta_a.to(device)
        meta_b = meta_b.to(device)
        target = target.to(device).squeeze(-1)  # Ensure target is (B,)

        optimizer.zero_grad()

        pred_mu, pred_sigma = model(img, meta_a, meta_b)

        loss = criterion(pred_mu, pred_sigma, target)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img.size(0)

    if scheduler:
        scheduler.step()

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def eval_epoch(model, loader, criterion, device, scaler_target):
    """
    Executes one validation epoch.
    Calculates both the optimization loss (scaled) and the competition metric (unscaled).
    """
    model.eval()
    running_loss = 0.0

    # Lists to store unscaled predictions and targets for metric calculation
    all_mu_unscaled = []
    all_sigma_unscaled = []
    all_target_unscaled = []

    # Extract scaler stats for inverse transform
    # scaler.scale_ is the std, scaler.mean_ is the mean
    std = scaler_target.scale_[0]
    mean = scaler_target.mean_[0]

    with torch.no_grad():
        for img, meta_a, meta_b, target in loader:
            img = img.to(device)
            meta_a = meta_a.to(device)
            meta_b = meta_b.to(device)
            target_gpu = target.to(device).squeeze(-1)

            pred_mu, pred_sigma = model(img, meta_a, meta_b)

            # 1. Calculate Loss (on scaled values, same as training)
            loss = criterion(pred_mu, pred_sigma, target_gpu)
            running_loss += loss.item() * img.size(0)

            # 2. Inverse Transform for Metric Calculation
            # Move to CPU numpy
            mu_np = pred_mu.cpu().numpy()
            sigma_np = pred_sigma.cpu().numpy()
            target_np = target_gpu.cpu().numpy()

            # Inverse transform
            # mu_ml = mu_scaled * std + mean
            mu_unscaled = mu_np * std + mean
            # sigma_ml = sigma_scaled * std (Scale only)
            sigma_unscaled = sigma_np * std
            # target_ml = target_scaled * std + mean
            target_unscaled = target_np * std + mean

            all_mu_unscaled.append(mu_unscaled)
            all_sigma_unscaled.append(sigma_unscaled)
            all_target_unscaled.append(target_unscaled)

    epoch_loss = running_loss / len(loader.dataset)

    # Concatenate all batches
    y_true = np.concatenate(all_target_unscaled)
    y_pred_mean = np.concatenate(all_mu_unscaled)
    y_pred_sigma = np.concatenate(all_sigma_unscaled)

    # Calculate competition metric (higher is better, usually negative)
    score = metric_score(y_true, y_pred_mean, y_pred_sigma)

    return epoch_loss, score


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main function to orchestrate the training process.
    """
    seed_everything(Config.SEED)

    # 1. Load Data
    print(f"Loading data (Debug={debug})...")
    train_ds, val_ds, test_ds, processor = get_data(debug=debug)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model
    print("Initializing CI-OP-DS Net...")
    device = torch.device(Config.DEVICE)
    model = CI_OP_DS_Net().to(device)

    # 3. Setup Optimizer with Differential Learning Rates
    # Backbone gets LR_BACKBONE, Heads get LR_HEAD
    backbone_params = list(model.backbone.parameters())

    # Collect all other parameters (Stream A, Stream B MLP, Head)
    head_params = []
    head_params += list(model.stream_a.parameters())
    head_params += list(model.stream_b_mlp.parameters())
    head_params += list(model.head.parameters())

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": Config.LR_BACKBONE},
            {"params": head_params, "lr": Config.LR_HEAD},
        ],
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Loss Function
    criterion = SmoothLaplaceLoss().to(device)

    # Scaler for inverse transform during validation
    scaler_target = processor.scalers["target_fvc"]

    # 4. Training Loop
    print("Starting training...")
    best_score = -float("inf")
    best_epoch = 0
    patience = 10  # Early stopping patience
    patience_counter = 0

    for epoch in range(epochs):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scheduler
        )

        # Validate
        val_loss, val_score = eval_epoch(
            model, val_loader, criterion, device, scaler_target
        )

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val Metric: {val_score:.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_score > best_score:
            print(
                f"  >>> Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving model..."
            )
            best_score = val_score
            best_epoch = epoch
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Metric: {best_score:.6f} at Epoch {best_epoch+1}")

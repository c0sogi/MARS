import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.utils import seed_everything, score
from library.data import LungDataset, get_transforms
from library.model import BBSLNet


class LaplaceLoss(nn.Module):
    """
    Implements the negative Modified Laplace Log Likelihood loss.

    Metric:
        sigma_clipped = max(sigma, 70)
        delta = min(|FVC_true - FVC_pred|, 1000)
        metric = - (sqrt(2) * delta) / sigma_clipped - ln(sqrt(2) * sigma_clipped)

    Loss = -Metric
    """

    def __init__(self):
        super().__init__()
        self.max_error = float(Config.MAX_ERROR)
        self.min_confidence = float(Config.MIN_CONFIDENCE)
        self.sqrt_2 = torch.sqrt(torch.tensor(2.0))

    def forward(self, outputs, targets, delta_weeks, baseline_fvcs):
        """
        Args:
            outputs: (B, 3) -> [alpha, sigma_base, sigma_growth]
            targets: (B,) -> True FVC
            delta_weeks: (B,) -> Weeks - Baseline_Week
            baseline_fvcs: (B,) -> Baseline FVC
        """
        # 1. Parse Model Outputs
        alpha = outputs[:, 0]
        sigma_base = outputs[:, 1]
        sigma_growth = outputs[:, 2]

        # 2. Calculate Predictions
        # FVC_pred = Baseline + alpha * delta_t
        fvc_pred = baseline_fvcs + alpha * delta_weeks

        # Sigma_pred = Sigma_base + Sigma_growth * |delta_t|
        sigma_pred = sigma_base + sigma_growth * torch.abs(delta_weeks)

        # 3. Apply Metric Constraints
        # Clip confidence
        sigma_clipped = torch.clamp(sigma_pred, min=self.min_confidence)

        # Calculate absolute error
        abs_error = torch.abs(targets - fvc_pred)

        # Clip error (delta)
        delta = torch.clamp(abs_error, max=self.max_error)

        # 4. Compute Loss (Negative Metric)
        # Loss = (sqrt(2) * delta) / sigma_clipped + ln(sqrt(2) * sigma_clipped)
        term1 = (self.sqrt_2 * delta) / sigma_clipped
        term2 = torch.log(self.sqrt_2 * sigma_clipped)

        loss = term1 + term2

        return torch.mean(loss)


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move inputs to device
        img_ax = batch["img_ax"].to(device)
        img_cor = batch["img_cor"].to(device)
        meta = batch["meta"].to(device)

        # Targets and Aux info
        targets = batch["fvc"].to(device)
        delta_weeks = batch["delta_week"].to(device)
        baseline_fvcs = batch["baseline_fvc"].to(device)

        optimizer.zero_grad()

        # Forward pass
        # outputs: [alpha, sigma_base, sigma_growth]
        outputs = model(img_ax, img_cor, meta)

        # Compute loss
        loss = criterion(outputs, targets, delta_weeks, baseline_fvcs)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_ax.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the official metric.
    """
    model.eval()

    all_true_fvc = []
    all_pred_fvc = []
    all_pred_sigma = []

    with torch.no_grad():
        for batch in loader:
            img_ax = batch["img_ax"].to(device)
            img_cor = batch["img_cor"].to(device)
            meta = batch["meta"].to(device)

            targets = batch["fvc"].cpu().numpy()
            delta_weeks = batch["delta_week"].to(device)
            baseline_fvcs = batch["baseline_fvc"].to(device)

            # Forward pass
            outputs = model(img_ax, img_cor, meta)

            alpha = outputs[:, 0]
            sigma_base = outputs[:, 1]
            sigma_growth = outputs[:, 2]

            # Calculate predictions
            fvc_pred = baseline_fvcs + alpha * delta_weeks
            sigma_pred = sigma_base + sigma_growth * torch.abs(delta_weeks)

            # Store results
            all_true_fvc.extend(targets)
            all_pred_fvc.extend(fvc_pred.cpu().numpy())
            all_pred_sigma.extend(sigma_pred.cpu().numpy())

    # Calculate official metric
    metric_score = score(all_true_fvc, all_pred_fvc, all_pred_sigma)
    return metric_score


def run_training(debug=False):
    """
    Main orchestration function for training.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # 2. Data Preparation
    print("Initializing Datasets...")
    train_dataset = LungDataset(
        csv_path=Config.TRAIN_CSV,
        mode="train",
        transform=get_transforms("train"),
        debug=debug,
    )

    val_dataset = LungDataset(
        csv_path=Config.VAL_CSV,
        mode="val",
        transform=get_transforms("val"),
        debug=debug,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # 3. Model & Optimizer
    print("Initializing Model...")
    model = BBSLNet().to(device)

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = LaplaceLoss().to(device)

    # 4. Training Loop
    best_score = -float("inf")
    patience_counter = 0

    epochs = 2 if debug else Config.EPOCHS

    print("Starting Training...")
    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Score: {val_score:.6f}"
        )

        # Checkpointing & Early Stopping
        if val_score > best_score:
            print(
                f"Score Improved ({best_score:.6f} -> {val_score:.6f}). Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), Config.CHECKPOINT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    print(f"Training Complete. Best Validation Score: {best_score:.6f}")
    return best_score

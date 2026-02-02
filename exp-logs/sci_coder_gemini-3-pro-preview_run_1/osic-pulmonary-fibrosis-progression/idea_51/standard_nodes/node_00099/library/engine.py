import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.dataset import LungDataset
from library.model import NSLHN
from library.loss import LaplaceLikelihoodLoss


def get_dataloaders(
    debug=False, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
):
    """
    Initializes datasets and dataloaders for training and validation.
    """
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)

    if debug:
        print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} rows.")
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # Initialize Datasets
    train_dataset = LungDataset(train_df, mode="train")
    val_dataset = LungDataset(val_df, mode="val")

    # Initialize Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move data to device
        img_axial = batch["img_axial"].to(device)
        img_coronal = batch["img_coronal"].to(device)
        tabular = batch["tabular"].to(device)
        time_delta = batch["time_delta"].to(device)
        baseline_fvc = batch["baseline_fvc"].to(device)
        target = batch["target"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(img_axial, img_coronal, tabular)

        # Calculate loss
        loss = criterion(outputs, target, baseline_fvc, time_delta)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_axial.size(0)
        count += img_axial.size(0)

    return running_loss / count if count > 0 else 0.0


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    metric_values = []

    with torch.no_grad():
        for batch in loader:
            img_axial = batch["img_axial"].to(device)
            img_coronal = batch["img_coronal"].to(device)
            tabular = batch["tabular"].to(device)
            time_delta = batch["time_delta"].to(device)
            baseline_fvc = batch["baseline_fvc"].to(device)
            target = batch["target"].to(device)

            # Forward pass
            outputs = model(img_axial, img_coronal, tabular)

            # Reconstruct Predictions from Parameters
            # outputs: [alpha, sigma_base, sigma_growth]
            alpha = outputs[:, 0:1]
            sigma_base = outputs[:, 1:2]
            sigma_growth = outputs[:, 2:3]

            # FVC_pred = Baseline + alpha * dt
            fvc_pred = baseline_fvc + alpha * time_delta

            # Sigma_pred = Base + Growth * |dt|
            sigma_pred = sigma_base + sigma_growth * torch.abs(time_delta)

            # Calculate Metric
            # calculate_metric expects numpy arrays or tensors and returns mean score
            score = calculate_metric(target, fvc_pred, sigma_pred)
            metric_values.append(score)

    return np.mean(metric_values) if metric_values else -999.0


def run_training(debug=Config.DEBUG):
    """
    Main execution function for training the model.
    """
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_loader, val_loader = get_dataloaders(debug=debug)

    # 2. Initialize Model
    model = NSLHN().to(device)

    # 3. Setup Training Components
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    criterion = LaplaceLikelihoodLoss().to(device)

    # 4. Training Loop with Early Stopping
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Score: {val_score} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping Check
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  -> Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_score

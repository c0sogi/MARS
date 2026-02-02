import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import LungDataset
from library.model import ChannelAdaptiveDualAxisNet
from library.loss import LaplaceLogLikelihoodLoss


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move data to device
        axial = batch["axial"].to(device)
        coronal = batch["coronal"].to(device)
        tabular = batch["tabular"].to(device)

        target_fvc = batch["fvc"].to(device)
        base_fvc = batch["base_fvc"].to(device)
        week_delta = batch["week"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass: Get parameters
        alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

        # Reconstruct Predictions based on linear trajectory
        # FVC_pred = Baseline + alpha * delta_week
        pred_fvc = base_fvc + alpha * week_delta

        # Sigma_pred = Sigma_base + Sigma_growth * |delta_week|
        pred_sigma = sigma_base + sigma_growth * torch.abs(week_delta)

        # Calculate Loss
        loss = criterion(pred_fvc, pred_sigma, target_fvc)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * axial.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average metric score (Negative Loss).
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            axial = batch["axial"].to(device)
            coronal = batch["coronal"].to(device)
            tabular = batch["tabular"].to(device)

            target_fvc = batch["fvc"].to(device)
            base_fvc = batch["base_fvc"].to(device)
            week_delta = batch["week"].to(device)

            # Forward pass
            alpha, sigma_base, sigma_growth = model(axial, coronal, tabular)

            # Reconstruct Predictions
            pred_fvc = base_fvc + alpha * week_delta
            pred_sigma = sigma_base + sigma_growth * torch.abs(week_delta)

            # Calculate Loss (Negative Metric)
            loss = criterion(pred_fvc, pred_sigma, target_fvc)

            running_loss += loss.item() * axial.size(0)

    avg_loss = running_loss / len(loader.dataset)

    # The metric is the negative of the loss (since loss minimizes negative metric)
    # Metric = -Loss
    return -avg_loss


def train_model(train_df, val_df, debug=False):
    """
    Main function to train the model with Early Stopping.

    Args:
        train_df (pd.DataFrame): Training metadata.
        val_df (pd.DataFrame): Validation metadata.
        debug (bool): If True, subsets data for quick debugging.

    Returns:
        str: Path to the saved best model.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Debugging: Subset data if requested
    if debug:
        print(f"DEBUG Mode: Training on {Config.DEBUG_DATA_SIZE} samples.")
        train_df = train_df.iloc[: Config.DEBUG_DATA_SIZE].copy()
        val_df = val_df.iloc[: Config.DEBUG_DATA_SIZE].copy()

    # 2. DataLoaders
    train_dataset = LungDataset(train_df, mode="train")
    val_dataset = LungDataset(val_df, mode="val")

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

    # 3. Model Initialization
    model = ChannelAdaptiveDualAxisNet()
    model.to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    criterion = LaplaceLogLikelihoodLoss()

    # 5. Training Loop
    best_score = -float("inf")
    patience_counter = 0
    best_model_path = Config.MODEL_SAVE_PATH

    print(f"Starting training on device: {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = evaluate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Score: {val_score} | "
            f"LR: {optimizer.param_groups[0]['lr']}"
        )

        # Early Stopping & Checkpointing
        if val_score > best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with score: {best_score}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score}")
    return best_model_path

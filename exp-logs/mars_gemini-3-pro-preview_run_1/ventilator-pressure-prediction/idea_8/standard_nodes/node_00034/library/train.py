import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os

from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data_processing import prepare_data
from library.dataset import VentilatorDataset
from library.model import VentilatorNet


def masked_mae_loss(preds, targets, u_out):
    """
    Computes L1 loss masked by the inspiratory phase (u_out == 0).
    """
    # u_out is 0 for inspiration, 1 for expiration.
    # Mask = 1 where we want to calculate loss (inspiration)
    mask = 1.0 - u_out

    # Element-wise absolute error masked
    loss = torch.abs(preds - targets) * mask

    # Average over the valid elements (avoid div by zero)
    return loss.sum() / (mask.sum() + 1e-8)


def train_epoch(model, loader, optimizer, scheduler, device):
    """
    Runs one epoch of training.
    """
    model.train()
    total_loss = 0.0

    for X, y, u_out in loader:
        X, y, u_out = X.to(device), y.to(device), u_out.to(device)

        optimizer.zero_grad()

        preds = model(X)
        loss = masked_mae_loss(preds, y, u_out)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate_epoch(model, loader, device):
    """
    Runs validation on the validation set.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for X, y, u_out in loader:
            X, y, u_out = X.to(device), y.to(device), u_out.to(device)

            preds = model(X)

            # Compute Loss
            loss = masked_mae_loss(preds, y, u_out)
            total_loss += loss.item()

            # Compute Metric (MAE on inspiratory phase)
            mae = compute_metric(preds, y, u_out)
            total_mae += mae

            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, _, _ in loader:
            X = X.to(device)
            preds = model(X)
            # Move to CPU and numpy
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: (N_breaths, 80)
    return np.concatenate(all_preds, axis=0)


def run_training():
    """
    Main function to execute the training pipeline.
    """
    # 1. Setup
    seed_everything()
    device = get_device()

    print(f"Running experiment: {Config.EXP_ID}")
    print(f"Device: {device}")

    # 2. Data Loading
    # prepare_data handles caching internally
    train_x, train_y, val_x, val_y, test_x, test_ids = prepare_data()

    train_dataset = VentilatorDataset(train_x, train_y)
    val_dataset = VentilatorDataset(val_x, val_y)
    test_dataset = VentilatorDataset(test_x, is_test=True)

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    input_dim = train_x.shape[2]
    model = VentilatorNet(input_dim=input_dim).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    # 5. Training Loop
    best_mae = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, val_mae = validate_epoch(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val MAE: {val_mae}"
        )

        # Save best model
        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! MAE: {best_mae}")

    print(f"Training complete. Best Validation MAE: {best_mae}")

    # 6. Inference & Submission
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))

    print("Generating predictions on test set...")
    preds_matrix = inference(model, test_loader, device)

    # Flatten predictions to match the 1D structure of test_ids
    # preds_matrix shape is (N_breaths, 80), flattening it aligns with the time-step ordered test_ids
    preds_flat = preds_matrix.flatten()

    # Create submission DataFrame
    submission = pd.DataFrame({"id": test_ids, "pressure": preds_flat})

    # Save submission
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

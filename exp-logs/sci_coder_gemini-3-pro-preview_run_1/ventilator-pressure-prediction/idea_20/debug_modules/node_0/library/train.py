import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, compute_metric
from library.dataset import prepare_datasets
from library.model import CuratedIdentityNet
from library.loss import MaskedL1Loss


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, aux_weight):
    """
    Handles the training of a single epoch.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        x = batch["x"].to(device)
        static = batch["static"].to(device)
        u_out = batch["u_out"].to(device)
        y = batch["y"].to(device)

        optimizer.zero_grad()

        # Forward pass
        final_pred, aux_pred = model(x, static)

        # Compute Loss
        # Reshape preds to match y if necessary (N, L)
        final_pred = final_pred.squeeze(-1)
        if aux_pred is not None:
            aux_pred = aux_pred.squeeze(-1)

        loss_final = criterion(final_pred, y, u_out)

        loss = loss_final
        if aux_pred is not None:
            loss_aux = criterion(aux_pred, y, u_out)
            loss = loss_final + aux_weight * loss_aux

        # Backward pass
        loss.backward()

        # Strict Gradient Clipping
        nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer and Scheduler steps
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and MAE.
    """
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            static = batch["static"].to(device)
            u_out = batch["u_out"].to(device)
            y = batch["y"].to(device)

            # Forward pass (only need final prediction for validation metric)
            final_pred, _ = model(x, static)
            final_pred = final_pred.squeeze(-1)

            # Compute Loss (for monitoring)
            loss = criterion(final_pred, y, u_out)

            # Compute Metric (MAE on inspiratory phase)
            mae = compute_metric(final_pred, y, u_out)

            total_loss += loss.item()
            total_mae += mae
            num_batches += 1

    return total_loss / num_batches, total_mae / num_batches


def generate_submission(model, loader, test_ids, device, output_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            static = batch["static"].to(device)

            # Forward pass
            final_pred, _ = model(x, static)
            final_pred = final_pred.squeeze(-1)  # (Batch, Seq_Len)

            # Collect predictions
            predictions.append(final_pred.cpu().numpy())

    # Concatenate all batches: (N_test_breaths, Seq_Len)
    predictions = np.concatenate(predictions, axis=0)

    # Flatten predictions to 1D array
    predictions_flat = predictions.flatten()

    # Ensure test_ids are also flat
    test_ids_flat = test_ids.flatten()

    if len(predictions_flat) != len(test_ids_flat):
        raise ValueError(
            f"Shape mismatch: Preds {len(predictions_flat)} vs IDs {len(test_ids_flat)}"
        )

    # Create DataFrame
    submission_df = pd.DataFrame({"id": test_ids_flat, "pressure": predictions_flat})

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Main training pipeline.
    """
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Preparation
    print("Preparing datasets...")
    train_loader, val_loader, test_loader, test_ids = prepare_datasets(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = CuratedIdentityNet().to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    # Total steps = epochs * steps_per_epoch
    total_steps = epochs * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        total_steps=total_steps,
        pct_start=Config.PCT_START,
        div_factor=Config.DIV_FACTOR,
        final_div_factor=Config.FINAL_DIV_FACTOR,
    )

    criterion = MaskedL1Loss()

    # 5. Training Loop
    best_val_mae = float("inf")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Train
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            Config.AUX_LOSS_WEIGHT,
        )

        # Validate
        val_loss, val_mae = validate(model, val_loader, criterion, device)

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Checkpoint
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with MAE: {best_val_mae}")

    print("Training complete.")

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, test_ids, device, Config.SUBMISSION_PATH)

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed
from library.data import get_dataloaders
from library.model import EUGAT


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        batch = batch.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch)

        # Compute loss
        loss = criterion(outputs, batch.y)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def evaluate(model, loader, criterion, device, scaler=None):
    """
    Evaluates the model on a given dataset.
    Returns the average loss (on scaled data) and a dictionary of metrics (on original scale).
    """
    model.eval()
    running_loss = 0.0
    num_batches = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # Forward pass
            outputs = model(batch)

            # Compute loss on scaled data (for scheduler/early stopping consistency)
            loss = criterion(outputs, batch.y)
            running_loss += loss.item()
            num_batches += 1

            # Collect predictions and targets for metric calculation
            all_preds.append(outputs.cpu())
            all_targets.append(batch.y.cpu())

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0

    # Concatenate all batches
    if not all_preds:
        return avg_loss, {}

    pred_tensor = torch.cat(all_preds, dim=0)
    target_tensor = torch.cat(all_targets, dim=0)

    # Inverse transform to get original scale values if scaler is provided
    if scaler is not None:
        pred_orig = scaler.inverse_transform(pred_tensor)
        target_orig = scaler.inverse_transform(target_tensor)
    else:
        pred_orig = pred_tensor
        target_orig = target_tensor

    # Ensure non-negative values for RMSLE calculation (clipping at 0)
    pred_orig = torch.clamp(pred_orig, min=0.0)
    target_orig = torch.clamp(target_orig, min=0.0)

    # Calculate Column-wise RMSLE
    # log(x + 1)
    log_pred = torch.log1p(pred_orig)
    log_target = torch.log1p(target_orig)

    # Squared differences
    squared_log_diff = (log_pred - log_target) ** 2

    # Mean over samples for each column
    mse_log_per_col = torch.mean(squared_log_diff, dim=0)

    # Root
    rmsle_per_col = torch.sqrt(mse_log_per_col)

    # Mean RMSLE across columns
    mean_rmsle = torch.mean(rmsle_per_col).item()

    metrics = {
        "rmsle_mean": mean_rmsle,
        "rmsle_formation": rmsle_per_col[0].item(),
        "rmsle_bandgap": rmsle_per_col[1].item(),
    }

    return avg_loss, metrics


def run_training(
    num_epochs=Config.NUM_EPOCHS,
    batch_size=Config.BATCH_SIZE,
    learning_rate=Config.LEARNING_RATE,
    load_cached_data=True,
    patience=Config.PATIENCE,
):
    """
    Main function to run the training pipeline.
    """
    # 1. Set Seed
    set_seed(Config.RANDOM_SEED)

    # 2. Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 3. Data Loading
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, scaler = get_dataloaders(
        batch_size=batch_size, load_cached_data=load_cached_data
    )

    # 4. Model Initialization
    print("Initializing Model...")
    model = EUGAT().to(device)

    # 5. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
    )

    criterion = nn.MSELoss()

    # 6. Training Loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print("Starting Training...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device, scaler)

        # Update Scheduler
        scheduler.step(val_loss)

        # Print Metrics
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"Train Loss (MSE): {train_loss:.6f} | "
            f"Val Loss (MSE): {val_loss:.6f} | "
            f"Val RMSLE Mean: {val_metrics['rmsle_mean']:.8f} | "
            f"Time: {time.time() - epoch_start:.2f}s"
        )

        # Checkpointing and Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  -> New best model saved (Val Loss: {val_loss:.6f})")
        else:
            epochs_no_improve += 1
            print(f"  -> No improvement. Patience: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time:.2f}s")

    # 7. Generate Submission
    print("Generating submission...")

    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    predictions = []

    # Inference on test set
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            predictions.append(outputs.cpu())

    # Concatenate and inverse transform
    pred_tensor = torch.cat(predictions, dim=0)
    pred_orig = scaler.inverse_transform(pred_tensor)

    # Clamp to be safe (energies shouldn't be negative typically for formation, definitely not for bandgap)
    # Formation energy can theoretically be negative, but bandgap cannot.
    # However, looking at training data, formation energy min is 0.0.
    pred_orig = torch.clamp(pred_orig, min=0.0)

    pred_np = pred_orig.numpy()

    # Create submission DataFrame
    # We need the IDs from the test metadata to ensure correct order/mapping
    test_meta_df = pd.read_csv(Config.TEST_METADATA_PATH)

    submission_df = pd.DataFrame(
        {
            "id": test_meta_df["id"],
            "formation_energy_ev_natom": pred_np[:, 0],
            "bandgap_energy_ev": pred_np[:, 1],
        }
    )

    # Save submission
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Print head of submission for verification
    print("Submission head:")
    print(submission_df.head())

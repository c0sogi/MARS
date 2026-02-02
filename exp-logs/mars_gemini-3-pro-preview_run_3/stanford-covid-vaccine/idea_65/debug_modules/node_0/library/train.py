import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library.utils import seed_everything, MCRMSE
from library.data import get_dataloaders
from library.model import HC_WG_BiGRU


class MCRMSELoss(nn.Module):
    """
    Custom Loss function for MCRMSE (Mean Columnwise Root Mean Squared Error).
    Optimizes on all 5 target columns over the scored sequence length (first 68 bases).
    """

    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        # y_pred, y_true shape: (Batch, Seq_Len, 5)

        # 1. Slice to the scored sequence length (Config.PRED_LEN = 68)
        # We ignore predictions/targets beyond the scored length during training
        y_pred = y_pred[:, : Config.PRED_LEN, :]
        y_true = y_true[:, : Config.PRED_LEN, :]

        # 2. Calculate MSE per column (averaging over batch and sequence dims)
        mse = torch.mean((y_pred - y_true) ** 2, dim=(0, 1))

        # 3. Calculate RMSE per column
        rmse = torch.sqrt(mse)

        # 4. Average RMSE across all 5 columns
        loss = torch.mean(rmse)

        return loss


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_mask)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * features.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the official MCRMSE metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"]  # Keep on CPU

            outputs = model(features, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using the utility function
    # This function handles slicing to PRED_LEN and filtering for SCORED_INDICES
    score = MCRMSE(all_targets, all_preds)
    return score


def generate_submission(model, loader, device):
    """
    Generates predictions for the test set and formats them for submission.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)

            outputs = model(features, pair_indices, pair_mask)
            all_preds.append(outputs.cpu().numpy())

    # Concatenate: (N_samples, 107, 5)
    preds = np.concatenate(all_preds, axis=0)

    # Load test metadata to get IDs
    test_df = pd.read_parquet(Config.TEST_METADATA)
    ids = test_df["id"].values

    # Prepare submission data
    submission_data = []

    # Iterate through samples and sequence positions
    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape: (107, 5)

        for seq_pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            row_values = sample_preds[seq_pos].tolist()
            submission_data.append([row_id] + row_values)

    columns = ["id_seqpos"] + Config.TARGET_COLS
    submission_df = pd.DataFrame(submission_data, columns=columns)
    return submission_df


def run_training():
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = Config.DEVICE
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Device: {device}")
    print(
        f"Model: HC-WG-BiGRU (Hidden: {Config.HIDDEN_DIM}, Layers: {Config.NUM_LAYERS})"
    )

    # 2. Data Loading
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = HC_WG_BiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience = 5
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score} | Time: {elapsed:.2f}s"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  >>> New Best Model Saved! Score: {best_score}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs without improvement."
            )
            break

    print(f"Training complete. Best Validation Score: {best_score}")

    # 6. Inference
    print("Generating submission...")
    # Load best model weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    submission_df = generate_submission(model, test_loader, device)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, calculate_metric
from library.data import get_loaders, load_data
from library.model import SDBR_BiGRU


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        features = batch["features"].to(device)
        pair_index = batch["pair_index"].to(device)
        targets = batch["targets"].to(device)  # (B, 68, 5)

        optimizer.zero_grad()

        # Forward pass: (B, 107, 5)
        preds = model(features, pair_index)

        # Slice predictions to match target length (68) for loss calculation
        preds_scored = preds[:, : Config.SEQ_SCORED, :]

        # Compute MCRMSE loss on all 5 targets
        loss = criterion(preds_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)
            targets = batch["targets"].to(device)

            preds = model(features, pair_index)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    val_preds = torch.cat(all_preds, dim=0)
    val_targets = torch.cat(all_targets, dim=0)

    # Calculate metric using the utility function
    # This handles slicing to seq_scored and filtering to the 3 scored columns
    score = calculate_metric(val_preds, val_targets)

    return score


def run_training(epochs=Config.EPOCHS, debug=False, load_cached_data=True, patience=5):
    """
    Main training pipeline with Early Stopping.
    """
    seed_everything(Config.SEED)

    # Load data
    train_loader, val_loader, _ = get_loaders(
        debug=debug, load_cached_data=load_cached_data
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SDBR_BiGRU().to(device)

    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = MCRMSELoss()

    best_score = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    counter = 0  # Early stopping counter

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()

        # Print metrics (Full precision for val_score)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

    print(f"Training finished. Best Val MCRMSE: {best_score}")
    return best_model_path


def predict_and_submit(model_path, debug=False, load_cached_data=True):
    """
    Generates predictions for the test set and creates a submission file.
    """
    # Load test data (dataframe needed for IDs)
    _, _, test_df = load_data(debug=debug, load_cached_data=load_cached_data)
    _, _, test_loader = get_loaders(debug=debug, load_cached_data=load_cached_data)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SDBR_BiGRU().to(device)

    # Load best model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            pair_index = batch["pair_index"].to(device)

            preds = model(features, pair_index)  # (B, 107, 5)
            all_preds.append(preds.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)  # (N_test, 107, 5)

    # Format Submission
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    ids = test_df["id"].values

    # Iterate through samples and sequence positions to create the submission format
    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for j, col in enumerate(target_cols):
                row_dict[col] = float(row_vals[j])
            submission_rows.append(row_dict)

    sub_df = pd.DataFrame(submission_rows)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

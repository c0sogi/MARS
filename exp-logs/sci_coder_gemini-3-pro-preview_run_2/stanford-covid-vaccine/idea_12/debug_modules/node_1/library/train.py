import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import set_seed, GlobalMCRMSE
from library.data import get_dataloaders, get_test_dataloader
from library.model import StackingDenseRefinedNet


def criterion_mcrmse(preds, targets, mask):
    """
    Calculates the Mean Columnwise Root Mean Squared Error (MCRMSE)
    strictly on the scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3).

    Args:
        preds: (B, L, 5)
        targets: (B, L, 5)
        mask: (B, L, 5)

    Returns:
        torch.Tensor: Scalar loss
    """
    # Indices corresponding to [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # based on Config.ALL_TARGETS order
    scored_indices = [0, 1, 3]

    loss = 0.0
    count_cols = 0

    mse = (preds - targets) ** 2

    for idx in scored_indices:
        # Extract specific column masks and errors
        m_col = mask[:, :, idx]
        mse_col = mse[:, :, idx]

        # Count valid positions
        n_valid = m_col.sum()

        if n_valid > 0:
            # RMSE for this column
            rmse = torch.sqrt((mse_col * m_col).sum() / n_valid)
            loss += rmse
            count_cols += 1

    if count_cols > 0:
        return loss / count_cols

    # Return zero loss with grad if no valid positions (edge case)
    return torch.tensor(0.0, device=preds.device, requires_grad=True)


def train_one_epoch(model, loader, optimizer, device):
    """
    Runs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, pair_indices, targets, mask) in enumerate(loader):
        features = features.to(device)
        pair_indices = pair_indices.to(device)
        targets = targets.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()

        preds = model(features, pair_indices)
        loss = criterion_mcrmse(preds, targets, mask)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Runs validation using GlobalMCRMSE to avoid batch-averaging bias.
    """
    model.eval()
    metric_calc = GlobalMCRMSE(scored_indices=[0, 1, 3], device=device)

    with torch.no_grad():
        for features, pair_indices, targets, mask in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)
            mask = mask.to(device)

            preds = model(features, pair_indices)
            metric_calc.update(preds, targets, mask)

    return metric_calc.compute()


def run_training(debug_size=None, epochs=Config.EPOCHS):
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Get Dataloaders
    train_loader, val_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug_size=debug_size,
    )

    # Initialize Model
    model = StackingDenseRefinedNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    best_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_loss}"
        )

        scheduler.step(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  Saved best model to {best_model_path}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Val MCRMSE: {best_loss}")


def generate_submission():
    """
    Generates submission file using the best trained model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found. Cannot generate submission.")
        return

    print(f"Loading model from {model_path}...")
    model = StackingDenseRefinedNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    test_loader, test_ids = get_test_dataloader(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    all_preds = []

    print("Running inference on test set...")
    with torch.no_grad():
        for features, pair_indices, _, _ in test_loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)

            preds = model(features, pair_indices)
            # Move to CPU and convert to numpy
            all_preds.append(preds.cpu().numpy())

    # Concatenate all batches: (N_samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Flatten to submission format
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    for i, sample_id in enumerate(test_ids):
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            # Get the 5 predictions for this position
            vals = all_preds[i, j]

            row = [row_id] + vals.tolist()
            submission_rows.append(row)

    columns = ["id_seqpos"] + Config.ALL_TARGETS
    sub_df = pd.DataFrame(submission_rows, columns=columns)

    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn

from library.config import (
    device,
    WORKING_DIR,
    SUBMISSION_PATH,
    SCORED_LEN,
    SEQ_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
    PATIENCE,
    SEED,
)
from library.utils import mcrmse_loss, seed_everything
from library.model import SSRFN
from library.data import get_dataloaders


def train_one_epoch(model, loader, optimizer, device):
    """
    Trains the model for one epoch using the masked loss strategy.

    Args:
        model: The PyTorch model.
        loader: DataLoader for training data.
        optimizer: The optimizer.
        device: Computation device.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        # Move inputs and targets to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # In training mode, SSRFN returns (y2, y1) for iterative refinement supervision
        preds_2, preds_1 = model(inputs)

        # Calculate losses strictly on the scored length (0-67)
        loss_2 = mcrmse_loss(preds_2, targets, mask_len=SCORED_LEN)
        loss_1 = mcrmse_loss(preds_1, targets, mask_len=SCORED_LEN)

        # Combined loss
        loss = loss_2 + 0.5 * loss_1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, device):
    """
    Validates the model by computing the Correct Global MCRMSE.
    Accumulates all predictions first to avoid batch-averaging bias.

    Args:
        model: The PyTorch model.
        loader: DataLoader for validation data.
        device: Computation device.

    Returns:
        float: The global MCRMSE score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # In eval mode, SSRFN returns only the final prediction y2
            preds = model(inputs)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate to form global arrays (N, L, 5)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Slice to valid scored length (0-67)
    valid_preds = all_preds[:, :SCORED_LEN, :]
    valid_targets = all_targets[:, :SCORED_LEN, :]

    # Select only the scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    p = valid_preds[:, :, scored_indices]
    t = valid_targets[:, :, scored_indices]

    # Compute MSE per column (averaging over all samples and positions)
    # axis=(0, 1) collapses the Batch and Length dimensions
    mse = np.mean((p - t) ** 2, axis=(0, 1))

    # Compute RMSE per column
    rmse = np.sqrt(mse)

    # Compute MCRMSE (Mean of RMSEs)
    score = np.mean(rmse)

    return score


def fit(
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LR,
    patience=PATIENCE,
    working_dir=WORKING_DIR,
    load_cached_data=True,
):
    """
    Orchestrates the training process including data loading, training loop,
    validation, early stopping, and model saving.

    Args:
        epochs (int): Maximum number of epochs.
        batch_size (int): Batch size.
        lr (float): Learning rate.
        patience (int): Patience for early stopping.
        working_dir (str): Directory for caching and saving models.
        load_cached_data (bool): Whether to use cached data.

    Returns:
        tuple: (Path to best model, Test DataLoader)
    """
    seed_everything(SEED)

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        working_dir=working_dir,
        batch_size=batch_size,
        load_cached_data=load_cached_data,
    )

    # 2. Initialize Model
    model = SSRFN().to(device)

    # 3. Setup Optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_score = float("inf")
    best_model_path = os.path.join(working_dir, "best_model.pth")
    patience_counter = 0

    print(f"Starting training on device: {device}")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_score = validate(model, val_loader, device)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        scheduler.step(val_score)

        # Checkpoint and Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val MCRMSE: {best_score}")
    return best_model_path, test_loader


def inference(model_path, test_loader, submission_path=SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves the submission CSV.

    Args:
        model_path (str): Path to the saved model weights.
        test_loader (DataLoader): DataLoader for the test set.
        submission_path (str): Path to save the submission file.
    """
    print("Generating submission...")

    # Load Model
    model = SSRFN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    submission_lines = []
    header = "id_seqpos,reactivity,deg_Mg_pH10,deg_pH10,deg_Mg_50C,deg_50C"
    submission_lines.append(header)

    # Access IDs attached to the dataset in library.data
    ids = test_loader.dataset.ids
    idx_counter = 0

    with torch.no_grad():
        for inputs in test_loader:
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Predict (returns B, L, 5)
            preds = model(inputs)
            preds_np = preds.cpu().numpy()

            batch_size = preds_np.shape[0]

            for i in range(batch_size):
                sample_id = ids[idx_counter]
                sample_preds = preds_np[i]  # Shape: (107, 5)

                # Generate a row for every sequence position (0 to 106)
                for seqpos in range(SEQ_LEN):
                    row_id = f"{sample_id}_{seqpos}"
                    vals = sample_preds[seqpos]

                    # Columns match model output order:
                    # reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
                    line = f"{row_id},{vals[0]},{vals[1]},{vals[2]},{vals[3]},{vals[4]}"
                    submission_lines.append(line)

                idx_counter += 1

    # Save to file
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    with open(submission_path, "w") as f:
        f.write("\n".join(submission_lines))

    print(f"Submission saved to {submission_path}")

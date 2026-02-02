import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import calculate_mcrmse, seed_everything


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for training data.
        optimizer: The optimizer.
        device: The device (cpu or cuda).
        epoch: Current epoch number (for logging).

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    total_batches = 0

    # MSE Loss with reduction='none' to handle masking manually
    criterion = nn.MSELoss(reduction="none")

    for batch in dataloader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)  # (B, 107, 3)
        mask = batch["mask"].to(device)  # (B, 107)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(sequence, loop_type, pair_dist)  # (B, 107, 3)

        # Calculate Loss
        # We only care about the scored positions defined by the mask
        raw_loss = criterion(outputs, targets)  # (B, 107, 3)

        # Apply mask: mask is (B, 107), unsqueeze to (B, 107, 1) to broadcast over channels
        masked_loss = raw_loss * mask.unsqueeze(-1)

        # Normalize loss: Sum of errors / Number of valid elements
        # Valid elements = sum(mask) * num_channels
        # We use a small epsilon to avoid division by zero (though unlikely with valid data)
        num_valid_elements = mask.sum() * Config.NUM_TARGETS
        loss = masked_loss.sum() / (num_valid_elements + 1e-8)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        total_batches += 1

    avg_loss = running_loss / total_batches if total_batches > 0 else 0.0
    print(f"Epoch {epoch+1} | Training Loss: {avg_loss:.6f}")

    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: DataLoader for validation data.
        device: The device.

    Returns:
        float: The MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            outputs = model(sequence, loop_type, pair_dist)

            # Extract only the scored positions (first 68) for metric calculation
            # Outputs/Targets shape: (B, 107, 3) -> Slice to (B, 68, 3)
            scored_preds = outputs[:, : Config.SCORED_LEN, :]
            scored_targets = targets[:, : Config.SCORED_LEN, :]

            all_preds.append(scored_preds.cpu().numpy())
            all_targets.append(scored_targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = calculate_mcrmse(y_true, y_pred)
    print(f"Validation MCRMSE: {score}")

    return score


def train_model(model, train_loader, val_loader, optimizer, device, scheduler=None):
    """
    Orchestrates the training process with Early Stopping and Model Checkpointing.

    Args:
        model: The PyTorch model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer.
        device: Device.
        scheduler: Learning rate scheduler (optional).
    """
    seed_everything(Config.SEED)

    best_mcrmse = float("inf")
    patience = 5
    counter = 0

    print(f"Starting training for {Config.EPOCHS} epochs on {device}...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Scheduler Step
        if scheduler is not None:
            scheduler.step()

        # Checkpointing and Early Stopping
        if val_mcrmse < best_mcrmse:
            print(
                f"Score Improved ({best_mcrmse} -> {val_mcrmse}). Saving model to {Config.MODEL_SAVE_PATH}"
            )
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            counter = 0
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{patience}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best MCRMSE: {best_mcrmse}")


def generate_submission(model, test_loader, device, save_path=Config.SUBMISSION_PATH):
    """
    Generates predictions for the test set and saves them to a CSV file.

    Args:
        model: The trained PyTorch model.
        test_loader: DataLoader for test data.
        device: Device.
        save_path: Path to save the submission CSV.
    """
    print("Generating submission...")

    # Load best model weights
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model weights from {Config.MODEL_SAVE_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using current model weights.")

    model.eval()
    model.to(device)

    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            batch_ids = batch["id"]  # List of IDs

            # Forward pass
            outputs = model(sequence, loop_type, pair_dist)  # (B, 107, 3)

            ids_list.extend(batch_ids)
            preds_list.append(outputs.cpu().numpy())

    # Concatenate predictions: (N_samples, 107, 3)
    preds_array = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    # We need to flatten the predictions to have one row per (id, seqpos)
    # The model predicts: reactivity, deg_Mg_pH10, deg_Mg_50C (indices 0, 1, 2)
    # Submission needs: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []
    seq_len = Config.SEQ_LEN

    # Efficiently construct the data
    # Flatten predictions to (N_samples * 107, 3)
    flat_preds = preds_array.reshape(-1, 3)

    # Create ID column
    # Repeat each ID seq_len times and append _0, _1, ...
    flat_ids = []
    for sample_id in ids_list:
        flat_ids.extend([f"{sample_id}_{i}" for i in range(seq_len)])

    # Create DataFrame
    df = pd.DataFrame(
        {
            "id_seqpos": flat_ids,
            "reactivity": flat_preds[:, 0],
            "deg_Mg_pH10": flat_preds[:, 1],
            "deg_pH10": 0.0,  # Not scored, fill with 0
            "deg_Mg_50C": flat_preds[:, 2],
            "deg_50C": 0.0,  # Not scored, fill with 0
        }
    )

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save
    df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

import os
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.config import Config
from library.loss import MCRMSELoss, mcrmse_metric
from library.model import StructuralBiGRU
from library.data_utils import get_dataloaders


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, pair_indices, targets) in enumerate(loader):
        inputs = inputs.to(device)
        pair_indices = pair_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Crucial for RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates all predictions before calculating the metric to avoid batch-averaging bias.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, pair_indices, targets in loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)
            targets = targets.to(device)

            outputs = model(inputs, pair_indices)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE on the specific scored columns (reactivity, deg_Mg_pH10, deg_Mg_50C)
    score = mcrmse_metric(all_preds, all_targets, scored_only=True)

    return score


def train_model(debug=False):
    """
    Main training loop with Early Stopping and Scheduler.
    """
    # Setup
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Data Loaders
    train_loader, val_loader, _ = get_dataloaders(debug=debug, load_cached_data=True)

    # Model
    model = StructuralBiGRU().to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    criterion = MCRMSELoss()

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Early Stopping variables
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Checkpoint & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved to {Config.BEST_MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training complete. Best Validation Score: {best_score}")


def predict_and_submit(model_path=None):
    """
    Generates predictions for the test set and creates the submission file.
    """
    if model_path is None:
        model_path = Config.BEST_MODEL_PATH

    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return

    device = torch.device(Config.DEVICE)

    # Load Data
    # Note: test_loader yields (inputs, pair_indices, ids)
    _, _, test_loader = get_dataloaders(debug=False, load_cached_data=True)

    # Load Model
    model = StructuralBiGRU().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    print("Generating predictions...")

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for inputs, pair_indices, ids in test_loader:
            inputs = inputs.to(device)
            pair_indices = pair_indices.to(device)

            # Forward pass: (Batch, 107, 5)
            outputs = model(inputs, pair_indices)
            outputs = outputs.cpu().numpy()

            preds_list.append(outputs)
            ids_list.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare submission data
    # We need to flatten the predictions to match the format: one row per seqpos
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_rows)

    # Save
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    # Example usage (commented out as per instructions)
    # train_model(debug=False)
    # predict_and_submit()
    pass

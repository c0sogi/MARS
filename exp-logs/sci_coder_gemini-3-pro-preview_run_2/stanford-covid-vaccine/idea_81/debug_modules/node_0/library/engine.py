import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_global_rmse
from library.data import get_dataloaders
from library.model import AHDRNModel


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for i, (inputs, partner_indices, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass: returns (y_2, y_1) in training mode
        y_2, y_1 = model(inputs, partner_indices)

        # Anchored Loss: Calculate MCRMSE on full sequence (0-107)
        loss_main = criterion(y_2, targets)
        loss_aux = criterion(y_1, targets)

        # Weighted sum
        loss = loss_main + Config.AUX_WEIGHT * loss_aux

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the official global RMSE metric.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, partner_indices, targets in dataloader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets_gpu = targets.to(device)

            # Forward pass: returns only y_2 in eval mode
            y_2 = model(inputs, partner_indices)

            # Calculate validation loss for monitoring (full sequence)
            loss = criterion(y_2, targets_gpu)
            running_loss += loss.item()

            # Collect for metric calculation
            all_preds.append(y_2.cpu().numpy())
            all_targets.append(targets.numpy())

    avg_loss = running_loss / len(dataloader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Compute official metric (sliced to 68, filtered columns)
    global_rmse, col_metrics = compute_global_rmse(all_preds, all_targets)

    return avg_loss, global_rmse, col_metrics


def train_model():
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = AHDRNModel().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # Loss
    criterion = MCRMSELoss()

    # Tracking
    best_metric = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric, col_metrics = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler based on validation metric
        scheduler.step(val_metric)

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_metric}"
        )

        # Early Stopping
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! (MCRMSE: {val_metric})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_metric}")
    return best_metric


def generate_submission():
    """
    Generates the submission file using the best trained model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Test Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = AHDRNModel().to(device)
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model not found at {Config.BEST_MODEL_PATH}")

    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    print("Generating predictions on test set...")
    all_preds = []

    with torch.no_grad():
        for inputs, partner_indices, _ in test_loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Inference (returns y_2)
            preds = model(inputs, partner_indices)
            all_preds.append(preds.cpu().numpy())

    # Shape: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load Test IDs to construct submission rows
    # We assume the dataloader preserves the order of test.csv (shuffle=False)
    test_df = pd.read_csv(Config.TEST_CSV)
    ids = test_df["id"].values

    # Prepare submission data
    submission_data = []

    # Iterate over each sample
    for i, sample_id in enumerate(ids):
        # Get predictions for this sample: (107, 5)
        sample_preds = all_preds[i]

        # Create a row for each sequence position
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            # Extract the 5 target values
            # Order: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            vals = sample_preds[seqpos]

            row_dict = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


# Note: No if __name__ == "__main__": block as requested.

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import set_seed, get_device, MCRMSELoss, format_submission
from library.data import get_dataloaders
from library.model import ConvTransformer


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Slice outputs and targets to scored sequence length (68)
        # The loss should only be calculated on the valid experimental data
        outputs_scored = outputs[:, : Config.PRED_LEN, :]
        targets_scored = targets[:, : Config.PRED_LEN, :]

        loss = criterion(outputs_scored, targets_scored)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Performs validation.
    """
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Slice for metric calculation
            outputs_scored = outputs[:, : Config.PRED_LEN, :]
            targets_scored = targets[:, : Config.PRED_LEN, :]

            loss = criterion(
                outputs_scored, targets_scored, col_indices=Config.SCORED_INDICES
            )

            running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def generate_predictions(model, dataloader, device):
    """
    Generates predictions for the test set.
    Returns full (N, 107, 5) predictions and list of IDs.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in dataloader:
            # Test loader returns inputs only, or we handle the tuple if dataset structure varies
            # Based on RNADataset implementation: if targets is None, it returns x
            # However, IDs are not returned by __getitem__, they are in the dataset object
            # We need to be careful. The provided RNADataset __getitem__ returns x (and y if exists).
            # It does NOT return IDs. We must align predictions with IDs from the dataset.

            # Since DataLoader shuffles=False for test, we can iterate sequentially.
            if isinstance(batch, (tuple, list)):
                inputs = batch[0]
            else:
                inputs = batch

            inputs = inputs.to(device)
            outputs = model(inputs)

            # Move to CPU and numpy
            all_preds.append(outputs.cpu().numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_preds, axis=0)

    # Get IDs from the dataset
    ids = dataloader.dataset.ids

    return predictions, ids


def run_training(debug=False):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = get_device()
    Config.create_dirs()

    print(f"Device: {device}")
    print(f"Debug Mode: {debug}")

    # 2. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=debug
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = ConvTransformer().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss} | "  # Full precision as requested
            f"LR: {current_lr:.2e} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"  New best model saved! (Val Loss: {best_val_loss})")
        else:
            patience_counter += 1
            print(
                f"  EarlyStopping counter: {patience_counter} out of {Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference and Submission
    print("\nGenerating submission...")

    # Load best model
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found, using current model state.")

    predictions, ids = generate_predictions(model, test_loader, device)

    # Format submission
    # predictions shape: (N, 107, 5)
    # format_submission handles flattening
    submission_df = format_submission(predictions, ids)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    # This block is technically not required by the prompt instructions ("Only implement the module class/functions"),
    # but provided for completeness if the file is executed directly.
    # The prompt specifically asked NOT to include the if __name__ == "__main__": block in the requirements,
    # but standard python modules often have it.
    # Re-reading requirements: "DO NOT include an if __name__ == "__main__": block."
    # I will remove this block in the final output.
    pass

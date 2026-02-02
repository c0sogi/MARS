import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.utils import seed_everything, get_device
from library.dataset import get_data_loaders
from library.model import HybridModel, train_epoch, validate_epoch


def train_model(
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 1e-3,
    debug: bool = False,
    patience: int = 15,
    save_dir: str = "./working/idea_3_execution",
):
    """
    Manages the training lifecycle of the Hybrid CNN-BiLSTM model.

    Args:
        epochs (int): Maximum number of training epochs.
        batch_size (int): Batch size for data loaders.
        lr (float): Learning rate.
        debug (bool): If True, uses a subset of data for quick debugging.
        patience (int): Number of epochs to wait for improvement before early stopping.
        save_dir (str): Directory to save model checkpoints and logs.
    """
    # 1. Initialization and Configuration
    seed_everything(42)
    device = get_device()
    os.makedirs(save_dir, exist_ok=True)

    print(f"Initializing training on device: {device}")

    # 2. Data Loading
    # Caching is handled internally by get_data_loaders in library/dataset.py
    train_loader, val_loader, test_loader = get_data_loaders(
        batch_size=batch_size, load_cached_data=True, debug=debug
    )

    # 3. Model Setup
    # Hyperparameters aligned with the provided Idea and Model definition
    model = HybridModel(
        input_dim=12, lstm_dim=512, num_lstm_layers=4, emb_dim=8, cnn_dim=256
    ).to(device)

    # 4. Optimization Setup
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    # Loss function: L1 Loss (MAE) over the entire breath sequence
    criterion = nn.L1Loss()

    # 5. Training Loop with Early Stopping
    best_mae = float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pth")
    early_stop_counter = 0

    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate (returns loss and inspiratory phase MAE)
        val_loss, val_mae = validate_epoch(model, val_loader, criterion, device)

        # Update Scheduler
        scheduler.step()

        # Print Metrics (Full Precision)
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val MAE: {val_mae}"
        )

        # Checkpointing
        if val_mae < best_mae:
            best_mae = val_mae
            early_stop_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1

        # Early Stopping
        if early_stop_counter >= patience:
            print(f"Early stopping triggered. No improvement for {patience} epochs.")
            break

    print(f"Training complete. Best Validation MAE: {best_mae}")

    # 6. Submission Generation
    print("Generating submission file...")

    # Load best model weights
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model weights.")

    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch in test_loader:
            X = batch["input"].to(device)
            # Model returns (Batch, Sequence_Length)
            preds = model(X)
            # Flatten to 1D array
            all_preds.append(preds.cpu().numpy().flatten())

    final_preds = np.concatenate(all_preds)

    # Map predictions to IDs using metadata
    test_meta_path = "./metadata/test_metadata.csv"
    if os.path.exists(test_meta_path):
        df_meta = pd.read_csv(test_meta_path)

        # Ensure metadata is sorted exactly as the dataset loader yields data
        # The dataset is sorted by breath_id, then time_step.
        # Assuming 'id' increases with time within a breath:
        df_meta = df_meta.sort_values(["breath_id", "id"])

        # Safety check for length mismatch (e.g., if debug mode was used)
        if len(final_preds) != len(df_meta):
            print(
                f"Warning: Prediction count ({len(final_preds)}) does not match metadata count ({len(df_meta)}). Truncating to minimum."
            )
            min_len = min(len(final_preds), len(df_meta))
            final_preds = final_preds[:min_len]
            df_meta = df_meta.iloc[:min_len]

        submission = pd.DataFrame({"id": df_meta["id"], "pressure": final_preds})

        # Save to ./submission/submission.csv as required
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        submission.to_csv(submission_path, index=False)
        print(f"Submission saved successfully to {submission_path}")

    else:
        print(
            f"Error: Test metadata file not found at {test_meta_path}. Cannot generate submission."
        )

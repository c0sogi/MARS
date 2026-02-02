import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from library.config import Config
from library.utils import set_seed, MCRMSELoss, compute_mcrmse
from library.data import get_dataloaders
from library.model import RNAGRUModel


def train_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    total_loss = 0.0
    num_samples = 0

    for features, targets in loader:
        features = features.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        # Output: (Batch, 107, 5)
        preds = model(features)

        # Slice predictions to match target length (68)
        # targets are already (Batch, 68, 5)
        preds_sliced = preds[:, : targets.shape[1], :]

        # Flatten for loss calculation
        # Treat (Batch * Seq_Len) as samples for the loss function
        preds_flat = preds_sliced.reshape(-1, preds_sliced.shape[-1])
        targets_flat = targets.reshape(-1, targets.shape[-1])

        loss = criterion(preds_flat, targets_flat)

        loss.backward()
        optimizer.step()

        # Accumulate weighted loss
        total_loss += loss.item() * features.size(0)
        num_samples += features.size(0)

    return total_loss / num_samples


def validate(model, loader, device, config):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score on the 3 scored columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    # Indices for scored columns: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    # Target cols: ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    scored_indices = [0, 1, 3]

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            # targets: (Batch, 68, 5)

            preds = model(features)
            # Slice to prediction length (68)
            preds_sliced = preds[:, : config.pred_len, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets.numpy())

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Filter for scored columns only
    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Compute metric
    score = compute_mcrmse(preds_scored, targets_scored)
    return score


def generate_submission(model, test_loader, config):
    """
    Generates submission file for the test set.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(config.device)
            preds = model(features)
            # Slice to prediction length (68)
            preds_sliced = preds[:, : config.pred_len, :]
            all_preds.append(preds_sliced.cpu().numpy())

    # Shape: (N_test, 68, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Load test metadata to get IDs
    test_df = pd.read_parquet(config.test_metadata_path)
    ids = test_df["id"].values

    # Prepare submission data
    submission_data = []
    target_cols = config.target_cols

    for i, sample_id in enumerate(ids):
        sample_preds = all_preds[i]  # (68, 5)
        for seqpos in range(config.pred_len):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_preds[col_idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def train_model(config=None):
    """
    Main training loop.
    """
    if config is None:
        config = Config()

    set_seed(config.seed)

    # Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Initialize Model
    model = RNAGRUModel(config).to(config.device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_max, eta_min=config.eta_min
    )

    # Loss Function
    criterion = MCRMSELoss()

    # Training State
    best_score = float("inf")
    patience = 10
    patience_counter = 0

    print(f"Starting training on {config.device}")
    print(config.get_config_info())

    for epoch in range(config.epochs):
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, config.device
        )

        # Validate
        val_score = validate(model, val_loader, config.device, config)

        # Update Scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE (Scored): {val_score:.6f} | LR: {current_lr:.2e}"
        )

        # Save Best Model & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved with score: {best_score:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs without improvement."
                )
                break

    print(f"Training complete. Best Validation Score: {best_score:.6f}")

    # Generate Submission with Best Model
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(config.model_save_path))
    generate_submission(model, test_loader, config)

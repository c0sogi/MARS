import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import Config
from library.utils import set_seed, mcrmse
from library.data import get_dataloaders
from library.model import GSRDN
from library.loss import MCRMSELoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, config):
    """
    Executes one training epoch using the Iterative Refinement strategy.

    Strategy:
    1. Compute static backbone features (Z) once.
    2. Pass 1: Predict using zero feedback -> Preds1.
    3. Pass 2: Predict using Graph-Smoothed feedback from detached Preds1 -> Preds2.
    4. Loss = Loss(Preds2) + 0.5 * Loss(Preds1).
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, pidx, targets) in enumerate(dataloader):
        features = features.to(device)
        pidx = pidx.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # 1. Static Features (Backbone runs once)
        z = model.get_static_features(features)

        # 2. Pass 1 (Zero Feedback)
        # prev_preds=None triggers zero initialization inside the model
        preds_1 = model(features, pidx, prev_preds=None, z_cached=z)

        # 3. Pass 2 (Feedback from Pass 1)
        # Detach gradients from Pass 1 as per strategy description
        preds_1_detached = preds_1.detach()
        preds_2 = model(features, pidx, prev_preds=preds_1_detached, z_cached=z)

        # 4. Loss Calculation
        # Criterion handles slicing to seq_scored internally
        loss_1 = criterion(preds_1, targets)
        loss_2 = criterion(preds_2, targets)

        # Weighted sum: L_total = L_pass2 + 0.5 * L_pass1
        loss = (config.loss_w_pass2 * loss_2) + (config.loss_w_pass1 * loss_1)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    return avg_loss


def validate(model, dataloader, criterion, device, config):
    """
    Validates the model using the two-pass inference strategy.
    Returns the average loss and the MCRMSE score on the validation set.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    # Indices for scoring (reactivity, deg_Mg_pH10, deg_Mg_50C)
    scored_indices = [config.target_cols.index(col) for col in config.scored_cols]

    with torch.no_grad():
        for features, pidx, targets in dataloader:
            features = features.to(device)
            pidx = pidx.to(device)
            targets = targets.to(device)

            # 1. Static Features
            z = model.get_static_features(features)

            # 2. Pass 1
            preds_1 = model(features, pidx, prev_preds=None, z_cached=z)

            # 3. Pass 2
            preds_2 = model(features, pidx, prev_preds=preds_1, z_cached=z)

            # Loss tracking (using combined loss for consistency with training)
            loss_1 = criterion(preds_1, targets)
            loss_2 = criterion(preds_2, targets)
            loss = (config.loss_w_pass2 * loss_2) + (config.loss_w_pass1 * loss_1)
            running_loss += loss.item()

            # Store predictions and targets for metric calculation
            # We only evaluate the final refined prediction (preds_2)
            # Slice to seq_scored length
            seq_scored = targets.shape[1]
            preds_cpu = preds_2[:, :seq_scored, :].cpu().numpy()
            targets_cpu = targets.cpu().numpy()

            # Select only scored columns
            all_preds.append(preds_cpu[:, :, scored_indices])
            all_targets.append(targets_cpu[:, :, scored_indices])

    avg_loss = running_loss / len(dataloader)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate MCRMSE
    score = mcrmse(all_targets, all_preds)

    return avg_loss, score


def run_training(debug=False, epochs=None, batch_size=None):
    """
    Main training routine.
    Initializes model, data, and optimizer, then runs the training loop with early stopping.
    """
    # 1. Setup
    config = Config(debug=debug)
    if epochs is not None:
        config.epochs = epochs
    if batch_size is not None:
        config.batch_size = batch_size

    set_seed(config.seed)

    print(f"Initializing training on device: {config.device}")
    print(
        f"Debug Mode: {debug}, Epochs: {config.epochs}, Batch Size: {config.batch_size}"
    )

    # 2. Data
    train_loader, val_loader, _ = get_dataloaders(debug=debug)

    # 3. Model
    model = GSRDN().to(config.device)

    # 4. Optimization
    criterion = MCRMSELoss().to(config.device)
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, verbose=True
    )

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    for epoch in range(config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, config.device, config
        )
        val_loss, val_score = validate(
            model, val_loader, criterion, config.device, config
        )

        scheduler.step(val_score)

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MCRMSE: {val_score:.20f}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  >>> New Best Model Saved (Score: {val_score:.6f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"  ... Patience {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best MCRMSE: {best_score}")
    return best_score


def generate_submission(model_path=None, debug=False):
    """
    Generates the submission file using the best trained model.
    Applies the two-pass inference strategy on the test set.
    """
    config = Config(debug=debug)
    device = config.device

    if model_path is None:
        model_path = config.model_save_path

    print(f"Generating submission using model: {model_path}")

    # Load Data
    _, _, test_loader = get_dataloaders(debug=debug)

    # Load Model
    model = GSRDN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Load Test Metadata to get IDs
    test_df = pd.read_csv(config.test_metadata_path)
    ids = test_df["id"].values

    all_preds = []

    with torch.no_grad():
        for features, pidx, _ in test_loader:
            features = features.to(device)
            pidx = pidx.to(device)

            # 1. Static Features
            z = model.get_static_features(features)

            # 2. Pass 1
            preds_1 = model(features, pidx, prev_preds=None, z_cached=z)

            # 3. Pass 2 (Final Prediction)
            preds_2 = model(features, pidx, prev_preds=preds_1, z_cached=z)

            # Move to CPU
            all_preds.append(preds_2.cpu().numpy())

    # Concatenate all predictions: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Ensure we have predictions for all IDs
    if len(all_preds) != len(ids):
        # If debug mode, we might have fewer predictions than total IDs if loader was truncated
        if not debug:
            print(f"Warning: Preds count {len(all_preds)} != IDs count {len(ids)}")

    # Prepare Submission Data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = (
        config.target_cols
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    print("Formatting submission rows...")
    for i, sample_id in enumerate(ids):
        if i >= len(all_preds):
            break

        sample_preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    submission_df = pd.DataFrame(submission_rows)

    # Save
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    # This block is provided for standalone testing if needed,
    # but the functions are designed to be imported.
    run_training(debug=True, epochs=2)
    generate_submission(debug=True)

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_metric
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import DeepStabilizedBiGRU


def train_and_predict(debug=Config.DEBUG, epochs=Config.MAX_EPOCHS):
    """
    Executes the training loop and generates the submission file.

    Args:
        debug (bool): If True, runs on a small subset of data for debugging.
        epochs (int): Maximum number of training epochs.
    """
    # 1. Initialization and Reproducibility
    seed_everything(42)
    device = torch.device(Config.DEVICE)

    # Ensure output directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"Initializing training on device: {device}")

    # 2. Data Loading
    # get_dataloaders handles caching and preprocessing internally
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model Setup
    model = DeepStabilizedBiGRU().to(device)

    # 4. Optimization
    # AdamW optimizer with Cosine Annealing Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 5. Loss Function
    # MCRMSELoss handles slicing to seq_scored (68) internally
    criterion = MCRMSELoss()

    # 6. Metric Configuration
    # We only score specific columns for validation (reactivity, deg_Mg_pH10, deg_Mg_50C)
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [target_cols.index(col) for col in scored_cols]

    # 7. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training loop...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            x, bppm, y, _ = batch
            x = x.to(device)
            bppm = bppm.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(x, bppm)

            # Loss calculation
            loss = criterion(preds, y)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Mandatory for stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD)

            # Optimization step
            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_preds_list = []
        val_targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                x, bppm, y, _ = batch
                x = x.to(device)
                bppm = bppm.to(device)

                preds = model(x, bppm)

                # Move to CPU for metric calculation to save GPU memory
                val_preds_list.append(preds.cpu())
                val_targets_list.append(y.cpu())

        # Concatenate all validation batches
        val_preds = torch.cat(val_preds_list, dim=0)
        val_targets = torch.cat(val_targets_list, dim=0)

        # Calculate MCRMSE only on scored columns
        val_mcrmse = calculate_metric(
            val_targets, val_preds, scored_cols_indices=scored_indices
        )

        # Update Scheduler
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # --- Early Stopping & Checkpointing ---
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved to {Config.MODEL_PATH}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 8. Inference and Submission Generation
    print("Generating submission...")

    # Load the best model state
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    test_preds_list = []
    test_ids_list = []

    with torch.no_grad():
        for batch in test_loader:
            x, bppm, _, ids = batch
            x = x.to(device)
            bppm = bppm.to(device)

            # Predict for full sequence length (107)
            preds = model(x, bppm)

            test_preds_list.append(preds.cpu().numpy())
            test_ids_list.extend(ids)

    # Concatenate all test predictions: Shape (N_test, 107, 5)
    all_preds = np.concatenate(test_preds_list, axis=0)

    # Format submission DataFrame
    # Rows: id_seqpos, Columns: targets
    submission_data = []
    target_cols = Config.TARGET_COLS

    for i, sample_id in enumerate(test_ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for t_idx, t_name in enumerate(target_cols):
                row_dict[t_name] = row_vals[t_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Save submission file
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

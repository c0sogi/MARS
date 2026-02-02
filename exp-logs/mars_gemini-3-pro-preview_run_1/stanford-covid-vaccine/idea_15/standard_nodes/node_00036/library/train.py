import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from library.config import Config
from library.utils import set_seed, mcrmse_metric, build_submission_df
from library.dataset import prepare_data
from library.model import RNAModel


def train_model():
    """
    Main function to train the Input-Injected Distance-Aware Residual BiGRU model.
    Handles data preparation, training loop, validation, early stopping, and inference.
    """
    # 1. Configuration and Setup
    config = Config()
    set_seed(config.SEED)

    # Ensure submission directory exists (redundant safety check)
    os.makedirs(os.path.dirname(config.SUBMISSION_PATH), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Preparation
    # Uses the caching mechanism implemented in library.dataset
    datasets = prepare_data(config, load_cached_data=True)

    train_loader = DataLoader(
        datasets["train"],
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = RNAModel(config).to(device)

    # 4. Optimizer and Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # 5. Loss Function
    # Standard MSE Loss. We will manually slice the output to the scored positions.
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_mcrmse = float("inf")
    patience = 0
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print("Starting training...")

    for epoch in range(config.EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

            optimizer.zero_grad()

            # Forward pass
            preds = model(seq, loop, dist)  # Shape: (B, 107, 3)

            # Slice predictions to match the scored sequence length (68)
            preds_scored = preds[:, : config.SEQ_SCORED, :]

            # Compute loss
            loss = criterion(preds_scored, targets)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Update scheduler
        scheduler.step()

        avg_train_loss = train_loss / len(train_loader)

        # --- Validation Phase ---
        model.eval()
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                seq = batch["seq"].to(device)
                loop = batch["loop"].to(device)
                dist = batch["dist"].to(device)
                targets = batch["targets"].to(device)

                preds = model(seq, loop, dist)
                preds_scored = preds[:, : config.SEQ_SCORED, :]

                all_val_preds.append(preds_scored.cpu())
                all_val_targets.append(targets.cpu())

        # Concatenate all batches
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)

        # Calculate Metric
        val_mcrmse = mcrmse_metric(all_val_targets, all_val_preds)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | Train MSE: {avg_train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # --- Early Stopping and Checkpointing ---
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            patience = 0
            # print(f"New best model saved to {best_model_path}")
        else:
            patience += 1
            if patience >= config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Best Val MCRMSE: {best_mcrmse}")

    # 7. Inference on Test Set
    print("Generating submission...")

    # Load best model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    else:
        print("Warning: Best model not found, using current model state.")

    model.eval()

    test_loader = DataLoader(
        datasets["test"],
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    all_test_preds = []
    all_test_ids = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            # Predict for full sequence length (107)
            preds = model(seq, loop, dist)  # Shape: (B, 107, 3)

            all_test_preds.append(preds.cpu())
            all_test_ids.extend(ids)

    all_test_preds = torch.cat(all_test_preds, dim=0)

    # 8. Format and Save Submission
    submission_df = build_submission_df(
        all_test_ids, all_test_preds, seq_len=config.SEQ_LENGTH
    )

    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

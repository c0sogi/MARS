import os
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import seed_everything, calculate_mcrmse, print_metric
from library.data import get_dataloaders
from library.model import RNAModel
from library.loss import MaskedMSELoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for batch in loader:
        # Move inputs to device
        sequence = batch["sequences"].to(device)
        loop_type = batch["loop_types"].to(device)
        pair_dist = batch["pair_dists"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(sequence, loop_type, pair_dist)

        # Calculate loss (MaskedMSELoss handles slicing internally)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * sequence.size(0)
        count += sequence.size(0)

    return running_loss / count if count > 0 else 0.0


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns average loss and MCRMSE.
    """
    model.eval()
    running_loss = 0.0
    count = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequences"].to(device)
            loop_type = batch["loop_types"].to(device)
            pair_dist = batch["pair_dists"].to(device)
            targets = batch["targets"].to(device)

            preds = model(sequence, loop_type, pair_dist)
            loss = criterion(preds, targets)

            running_loss += loss.item() * sequence.size(0)
            count += sequence.size(0)

            # Collect for metric calculation
            # We only score the first 68 positions for the metric
            # Targets are already shape (B, 68, 3)
            # Preds are (B, 107, 3), need slicing
            preds_sliced = preds[:, : Config.PRED_LEN, :]

            all_preds.append(preds_sliced.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    avg_loss = running_loss / count if count > 0 else 0.0

    # Concatenate all batches
    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Calculate Metric
    mcrmse = calculate_mcrmse(y_true, y_pred)

    return avg_loss, mcrmse


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for submission...")
    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequences"].to(device)
            loop_type = batch["loop_types"].to(device)
            pair_dist = batch["pair_dists"].to(device)
            batch_ids = batch["ids"]

            # Forward pass (B, 107, 3)
            preds = model(sequence, loop_type, pair_dist)

            ids_list.extend(batch_ids)
            preds_list.append(preds.cpu().numpy())

    # Concatenate all predictions: Shape (N_samples, 107, 3)
    all_preds = np.vstack(preds_list)

    # Prepare data for DataFrame
    submission_data = []

    # The 3 predicted columns
    # 0: reactivity, 1: deg_Mg_pH10, 2: deg_Mg_50C

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = float(sample_preds[seqpos, 0])
            deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Unscored columns set to 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": reactivity,
                    "deg_Mg_pH10": deg_Mg_pH10,
                    "deg_pH10": deg_pH10,
                    "deg_Mg_50C": deg_Mg_50C,
                    "deg_50C": deg_50C,
                }
            )

    # Create DataFrame
    df_sub = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = df_sub[cols]

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug_subset=None):
    """
    Main execution function.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
        debug_subset=debug_subset,
    )

    # 3. Model
    model = RNAModel(config=Config).to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 5. Loss
    criterion = MaskedMSELoss()

    # 6. Training Loop
    best_mcrmse = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Step Scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )  # Full precision printing

        # Save Best Model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! (MCRMSE: {best_mcrmse})")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse}")

    # 7. Generate Submission
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

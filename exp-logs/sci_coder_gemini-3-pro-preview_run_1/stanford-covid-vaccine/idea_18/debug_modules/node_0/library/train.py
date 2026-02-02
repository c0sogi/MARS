import os
import sys
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import Config
from library.dataset import get_dataset
from library.model import InterleavedBiGRU
from library.loss import MaskedMSELoss, mcrmse


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move batch to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        pair_dist = batch["pair_dist"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        preds = model(sequence, loop_type, pair_dist)

        # Calculate loss (Masked MSE)
        loss = criterion(preds, targets, mask)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_masks = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(sequence, loop_type, pair_dist)

            all_preds.append(preds)
            all_targets.append(targets)
            all_masks.append(mask)

    # Concatenate all batches
    if not all_preds:
        return 0.0

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    masks_cat = torch.cat(all_masks, dim=0)

    # Calculate MCRMSE
    score = mcrmse(preds_cat, targets_cat, masks_cat)
    return score.item()


def run_training():
    """
    Main execution function for training and submission generation.
    """
    # 1. Setup
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Load Data
    print("Loading datasets...")
    train_dataset = get_dataset("train", load_cached_data=True)
    val_dataset = get_dataset("val", load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    print("Initializing model...")
    model = InterleavedBiGRU().to(device)

    criterion = MaskedMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Cosine Annealing Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    best_mcrmse = float("inf")
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, Config.MAX_GRAD_NORM
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.EPOCHS} | "
            f"Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  [New Best Model Saved] MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            print(
                f"  [No Improvement] Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Inference and Submission
    print("\nStarting inference on test set...")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    test_dataset = get_dataset("test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Collect predictions
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            pair_dist = batch["pair_dist"].to(device)
            ids = batch["id"]

            # Predict (B, 107, 3)
            preds = model(sequence, loop_type, pair_dist)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 3)
    predictions = np.concatenate(all_preds, axis=0)

    # 6. Format Submission
    print("Formatting submission...")

    submission_rows = []

    # Iterate over each sample
    for i, sample_id in enumerate(all_ids):
        sample_preds = predictions[i]  # (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID: id_seqpos
            row_id = f"{sample_id}_{seqpos}"

            # Get predictions for this position
            # Scored targets: reactivity, deg_Mg_pH10, deg_Mg_50C
            p_reactivity = float(sample_preds[seqpos, 0])
            p_deg_Mg_pH10 = float(sample_preds[seqpos, 1])
            p_deg_Mg_50C = float(sample_preds[seqpos, 2])

            # Unscored targets: deg_pH10, deg_50C (Set to 0.0)
            p_deg_pH10 = 0.0
            p_deg_50C = 0.0

            submission_rows.append(
                {
                    "id_seqpos": row_id,
                    "reactivity": p_reactivity,
                    "deg_Mg_pH10": p_deg_Mg_pH10,
                    "deg_pH10": p_deg_pH10,
                    "deg_Mg_50C": p_deg_Mg_50C,
                    "deg_50C": p_deg_50C,
                }
            )

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)

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
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Total rows: {len(df_sub)}")


if __name__ == "__main__":
    run_training()

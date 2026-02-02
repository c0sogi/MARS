import os
import time
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, compute_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        # Forward pass
        optimizer.zero_grad()
        preds = model(features, pair_indices, pair_masks)

        # Compute loss
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score on the scoring columns.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"]  # Keep targets on CPU for concatenation later

            preds = model(features, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets)

    # Concatenate all batches
    if not all_preds:
        return float("inf")

    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metric
    score = compute_mcrmse(all_preds, all_targets)
    return score


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            preds = model(features, pair_indices, pair_masks)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, dim=0)
    return all_preds, all_ids


def generate_submission(preds, ids, output_path):
    """
    Formats predictions into the submission CSV format.

    Args:
        preds (np.ndarray): Shape (N, 107, 5)
        ids (list): List of sample IDs
        output_path (str): Path to save the CSV
    """
    # Columns as per sample_submission.csv
    # Config.TARGET_COLS matches the output order:
    # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    cols = Config.TARGET_COLS

    data_list = []

    for i, sample_id in enumerate(ids):
        sample_pred = preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(cols):
                row_dict[col_name] = row_values[col_idx]

            data_list.append(row_dict)

    df = pd.DataFrame(data_list)

    # Ensure column order matches sample submission
    output_cols = ["id_seqpos"] + cols
    df = df[output_cols]

    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    """
    Main execution function for training, validation, and submission generation.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = RNAModel().to(device)

    # 4. Optimization
    criterion = MCRMSELoss()
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.MAX_EPOCHS)

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training for {Config.MAX_EPOCHS} epochs...")

    for epoch in range(1, Config.MAX_EPOCHS + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch}/{Config.MAX_EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! (Score: {best_score:.10f})")
        else:
            patience_counter += 1
            print(
                f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    print("Generating predictions on test set...")
    test_preds, test_ids = inference(model, test_loader, device)

    # 7. Submission
    print("Saving submission...")
    generate_submission(test_preds, test_ids, Config.SUBMISSION_PATH)
    print("Done.")

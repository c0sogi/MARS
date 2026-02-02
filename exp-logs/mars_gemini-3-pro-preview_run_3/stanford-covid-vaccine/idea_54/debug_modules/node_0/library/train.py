import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from library
from library.config import (
    IDEA_DIR,
    SUBMISSION_DIR,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EPOCHS,
    PATIENCE,
    GRAD_CLIP,
    SEED,
    SEQ_LEN,
    set_seed,
    ALL_TARGETS,
)
from library.utils import compute_mcrmse, MCRMSELoss
from library.data import get_loaders
from library.model import DeepStabilizedBiGRU


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, pair_indices, pair_masks, targets) in enumerate(loader):
        features = features.to(device)
        pair_indices = pair_indices.to(device)
        pair_masks = pair_masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(features, pair_indices, pair_masks)

        # Compute loss (MCRMSE on all 5 targets)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns:
        val_loss: MCRMSE on all 5 targets (loss proxy).
        val_metric: MCRMSE on only the 3 scored targets (metric for selection).
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_indices, pair_masks, targets in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)
            targets = targets.to(device)

            preds = model(features, pair_indices, pair_masks)

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metrics
    # 1. Loss Proxy (All 5 targets)
    val_loss = compute_mcrmse(all_preds, all_targets, scoring_only=False).item()

    # 2. Selection Metric (Only 3 scored targets)
    val_metric = compute_mcrmse(all_preds, all_targets, scoring_only=True).item()

    return val_loss, val_metric


def generate_submission(model, test_loader, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []
    all_ids = []

    # Collect predictions
    with torch.no_grad():
        # The test loader in library.data returns (features, pair_indices, pair_masks, targets)
        # targets are dummy zeros for test set, but we need the IDs which are stored in the dataset
        # We access IDs directly from the dataset via the loader

        # Since the loader batches data, we need to iterate carefully.
        # The RNADataset stores IDs.

        # Iterate over loader
        start_idx = 0
        dataset_ids = test_loader.dataset.ids

        for features, pair_indices, pair_masks, _ in test_loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)

            preds = model(features, pair_indices, pair_masks)

            # Move to CPU numpy
            preds_np = preds.cpu().numpy()  # (B, 107, 5)
            all_preds.append(preds_np)

            batch_size = features.size(0)
            batch_ids = dataset_ids[start_idx : start_idx + batch_size]
            all_ids.extend(batch_ids)
            start_idx += batch_size

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, 107, 5)

    # Format for submission
    # We need to flatten: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_data = []

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            # Values: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            # Corresponds to indices 0, 1, 2, 3, 4
            row_values = sample_preds[seqpos].tolist()
            submission_data.append([row_id] + row_values)

    columns = ["id_seqpos"] + ALL_TARGETS
    submission_df = pd.DataFrame(submission_data, columns=columns)

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def run_training():
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 2. Model
    model = DeepStabilizedBiGRU().to(device)

    # 3. Optimization
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    best_metric = float("inf")
    best_model_path = os.path.join(IDEA_DIR, "best_model.pth")
    patience_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_metric = validate(model, val_loader, device)

        # Update scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss (All): {val_loss} | "
            f"Val Metric (Scored): {val_metric}"
        )

        # Early Stopping Check
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Metric: {best_metric}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # 5. Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    run_training()

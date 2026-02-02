import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import seed_everything, MCRMSELoss, global_mcrmse
from library.data import get_loaders
from library.model import CGSRBiGRU


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        features, adjacency, targets, masks = batch

        features = features.to(device)
        adjacency = adjacency.to(device)
        targets = targets.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(features, adjacency)

        # Apply mask to predictions and targets
        # We only care about positions where mask == 1
        # Flatten for loss calculation
        mask_bool = masks > 0.5
        preds_masked = preds[mask_bool]
        targets_masked = targets[mask_bool]

        loss = criterion(preds_masked, targets_masked)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features, adjacency, targets, masks = batch

            features = features.to(device)
            adjacency = adjacency.to(device)
            targets = targets.to(device)
            masks = masks.to(device)

            preds = model(features, adjacency)

            # For validation metric, we need to filter by the mask as well
            # to match the competition scoring (first 68 bases)
            mask_bool = masks > 0.5

            # We collect the masked values to compute the global metric
            # Note: global_mcrmse expects flattened or consistent shapes.
            # Since masks can vary (though here they are fixed to 68),
            # flattening masked values is safe.
            preds_masked = preds[mask_bool]
            targets_masked = targets[mask_bool]

            all_preds.append(preds_masked)
            all_targets.append(targets_masked)

    score = global_mcrmse(all_preds, all_targets)
    return score


def generate_submission(model, loader, device):
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in loader:
            features, adjacency, batch_ids = batch
            features = features.to(device)
            adjacency = adjacency.to(device)

            # Forward pass: (Batch, Seq_Len, 5)
            preds = model(features, adjacency)
            preds = preds.cpu().numpy()

            ids_list.extend(batch_ids)
            preds_list.append(preds)

    # Concatenate all predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(preds_list, axis=0)

    # Prepare data for DataFrame
    submission_data = []
    target_cols = Config.target_cols

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 5)

        for seq_pos in range(Config.seq_len):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_data.append(row_dict)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Save
    os.makedirs(os.path.dirname(Config.submission_path), exist_ok=True)
    submission_df.to_csv(Config.submission_path, index=False)
    print(f"Submission saved to {Config.submission_path}")


def run_training():
    seed_everything(Config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # Initialize Model
    model = CGSRBiGRU().to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.epochs, eta_min=Config.min_lr
    )

    criterion = MCRMSELoss()

    best_score = float("inf")

    print("Starting training...")
    for epoch in range(Config.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"Epoch {epoch+1}/{Config.epochs} | LR: {current_lr:.6f} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score}"
        )

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.model_save_path)
            print(f"New best model saved with MCRMSE: {best_score}")

    print(f"Training complete. Best Validation Score: {best_score}")

    # Load best model for submission
    model.load_state_dict(torch.load(Config.model_save_path, map_location=device))
    generate_submission(model, test_loader, device)


if __name__ == "__main__":
    run_training()

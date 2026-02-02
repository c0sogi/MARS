import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import seed_everything, mcrmse_loss, mcrmse_metric
from library.data import get_dataloaders
from library.model import HC_DBR_BiGRU


def train_one_epoch(model, loader, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in loader:
        # Move data to device
        features = batch["features"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["target"].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(features, pair_indices, pair_mask)

        # Calculate loss (Multi-Task Learning on all 5 columns)
        loss = mcrmse_loss(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally before calculating the metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["target"].to(device)

            outputs = model(features, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Global Aggregation
    if not all_preds:
        return 0.0

    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Calculate Metric (Scored columns only, sliced to 68)
    metric = mcrmse_metric(global_preds, global_targets)
    return metric


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and creates the submission file.
    Predicts for all 107 positions as required by the format.
    """
    model.eval()
    results = []

    # Target columns in the order output by the model
    target_cols = Config.TARGET_COLS

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]

            # Forward pass: (Batch, 107, 5)
            outputs = model(features, pair_indices, pair_mask)
            outputs = outputs.cpu().numpy()

            # Iterate through batch
            for i, sample_id in enumerate(ids):
                sample_pred = outputs[i]  # (107, 5)

                # Create rows for each position (0 to 106)
                for seq_pos in range(Config.SEQ_LEN):
                    row_id = f"{sample_id}_{seq_pos}"
                    row_data = {
                        "id_seqpos": row_id,
                        "reactivity": sample_pred[seq_pos, 0],
                        "deg_Mg_pH10": sample_pred[seq_pos, 1],
                        "deg_pH10": sample_pred[seq_pos, 2],
                        "deg_Mg_50C": sample_pred[seq_pos, 3],
                        "deg_50C": sample_pred[seq_pos, 4],
                    }
                    results.append(row_data)

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure column order matches sample submission
    cols_order = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols_order]

    # Save
    print(f"Saving submission to {output_path}")
    submission_df.to_csv(output_path, index=False)


def run_training():
    """
    Main execution function for training and evaluation.
    """
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing Model...")
    model = HC_DBR_BiGRU().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        # Validate
        val_metric = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_metric} | "
            f"LR: {current_lr:.2e}"
        )

        # Early Stopping & Model Checkpointing
        if val_metric < best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New Best Model Saved! (Metric: {best_metric})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Final Inference
    print("\nLoading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)
    print("Process completed.")


if __name__ == "__main__":
    # This block is provided for testing purposes if run directly,
    # but the instructions ask to only implement the module functions.
    # The run_training function encapsulates the main logic.
    run_training()

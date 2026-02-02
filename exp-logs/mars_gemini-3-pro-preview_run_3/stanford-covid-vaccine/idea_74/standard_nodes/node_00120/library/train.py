import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

# Import from provided library files
from library.config import Config
from library.utils import set_seed, get_score
from library.loss import MCRMSELoss
from library.data import get_loaders
from library.model import HCTADPBiGRU


def train_one_epoch(model, loader, criterion, optimizer, device, max_grad_norm):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (features, pair_indices, pair_masks, targets, _) in enumerate(
        loader
    ):
        features = features.to(device)
        pair_indices = pair_indices.to(device)
        pair_masks = pair_masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, pair_indices, pair_masks)

        # Loss calculation
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping for stability (Lesson 46)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    """
    Evaluates the model on the validation set.
    Returns the average loss and the MCRMSE score.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, pair_indices, pair_masks, targets, _ in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)
            targets = targets.to(device)

            outputs = model(features, pair_indices, pair_masks)

            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Store predictions and targets for metric calculation
            # Move to CPU to save GPU memory
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate competition metric
    # get_score handles slicing and column selection internally
    mcrmse_score = get_score(all_targets, all_preds)

    return running_loss / len(loader), mcrmse_score


def generate_submission(model, loader, device, submission_path):
    """
    Generates predictions for the test set and saves the submission file.
    """
    model.eval()
    ids_list = []
    preds_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for features, pair_indices, pair_masks, _, sample_ids in loader:
            features = features.to(device)
            pair_indices = pair_indices.to(device)
            pair_masks = pair_masks.to(device)

            # Forward pass: (B, 107, 5)
            outputs = model(features, pair_indices, pair_masks)

            preds_list.append(outputs.cpu().numpy())
            ids_list.extend(sample_ids)

    # Concatenate predictions: (N_samples, 107, 5)
    preds_array = np.concatenate(preds_list, axis=0)

    # Prepare data for submission DataFrame
    # We need to flatten the predictions to have one row per (id, seqpos)
    # Target columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    target_cols = Config.target_columns

    submission_data = []

    for i, sample_id in enumerate(ids_list):
        # Retrieve prediction for this sample: (107, 5)
        sample_pred = preds_array[i]

        for seqpos in range(Config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_pred[seqpos].tolist()
            submission_data.append([row_id] + row_values)

    # Create DataFrame
    columns = ["id_seqpos"] + target_cols
    sub_df = pd.DataFrame(submission_data, columns=columns)

    # Save to CSV
    sub_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training():
    """
    Main execution function for the training pipeline.
    """
    # 1. Setup
    config = Config()
    set_seed(config.seed)
    device = torch.device(config.device)

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")

    # 2. Data Loading
    print("Preparing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model Initialization
    print("Initializing Model...")
    model = HCTADPBiGRU().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

    criterion = MCRMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(config.epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, config.max_grad_norm
        )

        # Validate
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"LR: {current_lr:.2e} | "
            f"Train Loss: {train_loss:.5f} | "
            f"Val Loss: {val_loss:.5f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), config.best_model_path)
            print(f"  [+] New Best Model Saved! Score: {best_mcrmse:.10f}")
        else:
            patience_counter += 1
            print(
                f"  [-] No improvement. Patience: {patience_counter}/{config.patience}"
            )

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("\nTraining Complete. Loading best model for inference...")
    model.load_state_dict(torch.load(config.best_model_path, map_location=device))

    generate_submission(model, test_loader, device, config.submission_path)


if __name__ == "__main__":
    run_training()

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import get_dataloaders, get_test_loader
from library.model import RNAModel
from library.loss import MCRMSELoss
from library.metrics import calculate_competition_metric


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, bpp_indices, bpp_masks, targets) in enumerate(loader):
        inputs = inputs.to(device)
        bpp_indices = bpp_indices.to(device)
        bpp_masks = bpp_masks.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, bpp_indices, bpp_masks)

        # Calculate loss (MCRMSE on all targets)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for deep RNN stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad_norm)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device, config):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, bpp_indices, bpp_masks, targets in loader:
            inputs = inputs.to(device)
            bpp_indices = bpp_indices.to(device)
            bpp_masks = bpp_masks.to(device)
            targets = targets.to(device)

            preds = model(inputs, bpp_indices, bpp_masks)

            # Move to CPU to save GPU memory
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate competition metric (MCRMSE on scored columns/positions only)
    score = calculate_competition_metric(all_preds, all_targets, config)
    return score.item()


def inference_and_submission(model, device, config):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")
    model.eval()
    test_loader = get_test_loader(config)

    all_preds = []
    all_ids = []

    # Inference Loop
    with torch.no_grad():
        for inputs, bpp_indices, bpp_masks, sample_ids in test_loader:
            inputs = inputs.to(device)
            bpp_indices = bpp_indices.to(device)
            bpp_masks = bpp_masks.to(device)

            preds = model(inputs, bpp_indices, bpp_masks)
            # Preds shape: (Batch, 107, 5)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(sample_ids)

    # Concatenate all predictions: (Total_Samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare submission data
    # We need to flatten: id_seqpos, and the 5 target columns
    submission_data = []
    target_cols = config.target_cols

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(config.seq_len):
            # Construct ID: e.g., id_00073f8be_0
            row_id = f"{sample_id}_{seqpos}"
            row_preds = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_preds[col_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Ensure correct column order
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    submission_df.to_csv(config.submission_file, index=False)
    print(f"Submission saved to {config.submission_file}")


def run_training(debug=False, num_epochs=None):
    """
    Main function to run the training pipeline.
    """
    # 1. Setup
    config = Config(debug=debug)
    if num_epochs is not None:
        config.num_epochs = num_epochs

    seed_everything(config.seed)
    device = get_device()

    print(f"Using device: {device}")
    print(
        f"Model: Deep Residual High-Capacity BiGRU (Layers={config.num_layers}, Hidden={config.hidden_dim})"
    )

    # 2. Data
    # Caching is handled internally by get_dataloaders -> load_or_process_data
    train_loader, val_loader = get_dataloaders(config)

    # 3. Model
    model = RNAModel(config).to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.num_epochs):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, config
        )

        # Validate
        val_score = validate(model, val_loader, device, config)

        # Step Scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print Metrics
        print(
            f"Epoch {epoch+1}/{config.num_epochs} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config.patience}")

        if patience_counter >= config.patience:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(config.model_save_path, map_location=device))

    inference_and_submission(model, device, config)

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random

# Import from provided library files
from library.config import Config
from library.data_utils import load_or_process_data
from library.model_components import RNAModel
from library.loss_metric import MCRMSELoss, calculate_mcrmse


def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_model(config):
    """
    Executes the training pipeline for the Deep Decoupled Post-Norm BiGRU.
    """
    # Set seed for reproducibility
    set_seed(config.seed)

    device = torch.device(config.device)

    # Load Datasets using the caching utility
    # load_or_process_data handles the logic of loading from .npz or processing from .parquet
    train_dataset = load_or_process_data("train", config, load_cached_data=True)
    val_dataset = load_or_process_data("val", config, load_cached_data=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Initialize Model
    print(f"Initializing RNAModel on {device}...")
    model = RNAModel(config).to(device)

    # Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs, eta_min=config.eta_min
    )

    # Loss Function (MCRMSE on all 5 targets for training signal)
    criterion = MCRMSELoss()

    # Training State
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(config.epochs):
        model.train()
        train_loss_accum = 0.0

        for batch in train_loader:
            inputs = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            targets = batch["targets"].to(device)

            optimizer.zero_grad()

            # Forward pass
            preds = model(inputs, bpp_indices, bpp_mask)

            # Loss calculation
            loss = criterion(preds, targets)

            # Backward pass
            loss.backward()

            # Gradient Clipping (Critical for deep RNN stability)
            nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

            optimizer.step()

            train_loss_accum += loss.item()

        avg_train_loss = train_loss_accum / len(train_loader)

        # Validation
        model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["sequence"].to(device)
                bpp_indices = batch["bpp_indices"].to(device)
                bpp_mask = batch["bpp_mask"].to(device)
                targets = batch["targets"]  # Keep on CPU for metric calculation

                preds = model(inputs, bpp_indices, bpp_mask)

                all_preds.append(preds.cpu())
                all_targets.append(targets)

        # Concatenate for global metric calculation
        all_preds_tensor = torch.cat(all_preds, dim=0)
        all_targets_tensor = torch.cat(all_targets, dim=0)

        # Calculate Competition Metric (MCRMSE on scored columns only)
        val_mcrmse = calculate_mcrmse(all_preds_tensor, all_targets_tensor)

        # Scheduler Step
        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{config.epochs} - Train Loss: {avg_train_loss} - Validation MCRMSE: {val_mcrmse}"
        )

        # Checkpointing & Early Stopping
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), config.model_save_path)
            print(f"New best model saved to {config.model_save_path}")
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training completed. Best Validation MCRMSE: {best_mcrmse}")
    return config.model_save_path


def generate_submission(config):
    """
    Generates predictions for the test set and creates the submission file.
    """
    set_seed(config.seed)
    device = torch.device(config.device)

    # Load Test Data
    test_dataset = load_or_process_data("test", config, load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # Load Model
    model = RNAModel(config).to(device)
    if os.path.exists(config.model_save_path):
        model.load_state_dict(torch.load(config.model_save_path, map_location=device))
        print(f"Loaded model from {config.model_save_path}")
    else:
        print(
            f"Warning: Model file not found at {config.model_save_path}. Using random initialization."
        )

    model.eval()

    all_preds = []
    all_ids = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            bpp_mask = batch["bpp_mask"].to(device)
            ids = batch["id"]

            # Predict
            preds = model(inputs, bpp_indices, bpp_mask)

            all_preds.append(preds.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions: (N_samples, 107, 5)
    all_preds_np = np.concatenate(all_preds, axis=0)

    # Prepare Submission DataFrame
    # Rows must be id_seqpos for all 107 positions
    submission_data = []
    target_cols = (
        config.target_cols
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds_np[i]  # Shape (107, 5)

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"
            row_vals = sample_preds[seqpos]

            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_vals[col_idx]

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Save Submission
    submission_df.to_csv(config.submission_path, index=False)
    print(f"Submission saved to {config.submission_path}")


def run_task():
    """
    Main entry point to execute the full task pipeline.
    """
    config = Config()

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # 1. Train the model
    train_model(config)

    # 2. Generate submission
    generate_submission(config)

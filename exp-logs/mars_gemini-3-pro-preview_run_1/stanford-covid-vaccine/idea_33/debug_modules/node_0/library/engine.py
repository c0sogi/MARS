import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from library.config import Config
from library.dataset import get_dataset
from library.model import RNAModel
from library.utils import seed_everything, mcrmse_loss, format_submission


def train_one_epoch(model, loader, optimizer, device, config):
    """
    Trains the model for one epoch using Masked MSE loss on scored positions.
    """
    model.train()
    total_loss = 0.0

    # Iterate over the dataloader
    for batch in loader:
        # Move batch data to the specified device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(batch)
        targets = batch["targets"]

        # Masked MSE: Calculate loss only on the first 68 (SEQ_SCORED) positions
        # preds shape: [Batch, Seq_Len, 3]
        preds_scored = preds[:, : config.SEQ_SCORED, :]
        targets_scored = targets[:, : config.SEQ_SCORED, :]

        # Compute Mean Squared Error
        loss = F.mse_loss(preds_scored, targets_scored)

        # Backpropagation
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Return average loss for the epoch
    return total_loss / len(loader)


def evaluate(model, loader, device, config):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            preds = model(batch)
            targets = batch["targets"]

            # Slice to scored region for evaluation
            preds_scored = preds[:, : config.SEQ_SCORED, :]
            targets_scored = targets[:, : config.SEQ_SCORED, :]

            all_preds.append(preds_scored.cpu())
            all_targets.append(targets_scored.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE using the utility function which averages RMSE per column
    score = mcrmse_loss(all_targets, all_preds).item()
    return score


def run_training(config=Config, debug=False, num_samples=None):
    """
    Main training loop.
    """
    # Set seeds for reproducibility
    seed_everything(config.SEED)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    print(f"Initializing training (Debug={debug})...")

    # Load Datasets
    # Pass debug/num_samples to get_dataset to handle subsetting if requested
    train_ds = get_dataset("train", config, debug=debug, num_samples=num_samples)
    val_ds = get_dataset("val", config, debug=debug, num_samples=num_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = RNAModel(config).to(config.DEVICE)

    # Optimizer: AdamW with low weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS
    )

    best_mcrmse = float("inf")
    best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {config.DEVICE} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, config.DEVICE, config
        )
        val_mcrmse = evaluate(model, val_loader, config.DEVICE, config)

        scheduler.step()

        # Print full precision metrics
        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.10f} | "
            f"Val MCRMSE: {val_mcrmse:.10f}"
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), best_model_path)
            print("  New best model saved!")

    print(f"Training complete. Best Val MCRMSE: {best_mcrmse:.10f}")
    return best_model_path


def run_inference(config=Config):
    """
    Runs inference on the test set and generates the submission file.
    """
    print("Generating submission...")
    seed_everything(config.SEED)

    # Load Test Data
    test_ds = get_dataset("test", config)
    test_loader = DataLoader(
        test_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = RNAModel(config).to(config.DEVICE)
    model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print(
            f"Error: Model file not found at {model_path}. Cannot generate submission."
        )
        return

    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
    model.eval()

    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in test_loader:
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(config.DEVICE)

            ids = batch["ids"]
            preds = model(batch)  # [B, 107, 3]

            all_preds.append(preds.cpu())
            all_ids.extend(ids)

    # Concatenate predictions: [N_test, 107, 3]
    if len(all_preds) > 0:
        all_preds = torch.cat(all_preds, dim=0)
    else:
        print("No predictions generated.")
        return

    # Format submission
    df_sub = format_submission(all_ids, all_preds, seq_length=config.SEQ_LENGTH)

    # Save
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

import os
import time
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import Config
from library.utils import set_seed, MetricCalculator
from library.data import get_dataloaders
from library.model import HighCapacityAugmentedBiGRU


def train_one_epoch(model, loader, optimizer, metric_calc, device, config):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move batch to device
        sequence = batch["sequence"].to(device)
        bpp_indices = batch["bpp_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        batch_size = sequence.size(0)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        predictions = model(sequence, bpp_indices, pair_mask)

        # Compute loss (MCRMSE on all 5 targets)
        loss = metric_calc.compute_train_loss(predictions, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, loader, metric_calc, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            # Forward pass
            predictions = model(sequence, bpp_indices, pair_mask)

            # Store predictions and targets
            all_preds.append(predictions.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metric (MCRMSE on 3 scored targets)
    score = metric_calc.compute_val_metric(all_preds, all_targets)
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
            sequence = batch["sequence"].to(device)
            bpp_indices = batch["bpp_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]

            # Forward pass
            predictions = model(sequence, bpp_indices, pair_mask)

            all_preds.append(predictions.cpu().numpy())
            all_ids.extend(ids)

    # Concatenate predictions
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def main():
    """
    Main execution function.
    """
    # Initialize configuration
    config = Config()

    # Set reproducibility
    set_seed(config.SEED)

    # Get DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(config)

    # Initialize Model
    model = HighCapacityAugmentedBiGRU(config)
    model.to(config.DEVICE)

    # Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)

    # Metric Calculator
    metric_calc = MetricCalculator(config)

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {config.DEVICE}...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, optimizer, metric_calc, config.DEVICE, config
        )

        # Validate
        val_score = validate(model, val_loader, metric_calc, config.DEVICE)

        # Scheduler Step
        scheduler.step()

        elapsed_time = time.time() - start_time

        # Print metrics (full precision for val_score)
        print(
            f"Epoch {epoch + 1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "
            f"Time: {elapsed_time:.2f}s"
        )

        # Early Stopping and Checkpointing
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    # Load Best Model for Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(config.MODEL_PATH, map_location=config.DEVICE))

    # Generate Predictions
    print("Generating submission...")
    preds, ids = inference(model, test_loader, config.DEVICE)

    # Format Submission
    submission_rows = []
    target_cols = config.TARGET_COLS

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape: (107, 5)

        for seqpos in range(config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create row: [id_seqpos, val1, val2, val3, val4, val5]
            submission_rows.append([row_id] + row_values)

    # Create DataFrame
    columns = ["id_seqpos"] + target_cols
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    # Save Submission
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

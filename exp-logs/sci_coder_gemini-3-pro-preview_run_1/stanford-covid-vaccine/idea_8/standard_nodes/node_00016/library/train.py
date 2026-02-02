import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import AverageMeter, set_seed, mcrmse
from library.data import get_dataloaders
from library.model import RNAModel


def train_epoch(model, loader, criterion, optimizer, device, config):
    """
    Trains the model for one epoch.
    """
    model.train()
    losses = AverageMeter()

    for batch in loader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        loop = batch["loop"].to(device)
        distance = batch["distance"].to(device)
        targets = batch["target"].to(device)  # Shape: (B, 68, 5)

        # Forward pass
        # Output shape: (B, 107, 5)
        outputs = model(sequence, loop, distance)

        # Slice outputs to match target length (first 68 positions)
        # We only train on the positions where we have ground truth
        outputs_scored = outputs[:, : config.PRED_LEN, :]

        # Filter for scored columns only (Cite debug_lesson_4)
        outputs_scored = outputs_scored[:, :, config.SCORED_INDICES]
        targets_scored = targets[:, :, config.SCORED_INDICES]

        # Compute loss
        loss = criterion(outputs_scored, targets_scored)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

        optimizer.step()

        losses.update(loss.item(), sequence.size(0))

    return losses.avg


def validate(model, loader, device, config):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            distance = batch["distance"].to(device)
            targets = batch["target"].to(device)

            # Forward pass
            outputs = model(sequence, loop, distance)

            # Slice outputs to match target length
            outputs_scored = outputs[:, : config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)  # (N, 68, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 68, 5)

    # Calculate MCRMSE
    # We only score on specific columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    # Indices in TARGET_COLS: 0 (reactivity), 1 (deg_Mg_pH10), 3 (deg_Mg_50C)
    # The task description says: "While the submission format requires all 5 to be predicted,
    # only the following are scored: reactivity, deg_Mg_pH10, and deg_Mg_50C."

    # Filter for scored columns only (Cite debug_lesson_4)
    y_pred_scored = y_pred[:, :, config.SCORED_INDICES]
    y_true_scored = y_true[:, :, config.SCORED_INDICES]

    score = mcrmse(y_true_scored, y_pred_scored)

    return score


def train_model(config=Config):
    """
    Main training loop with Early Stopping.
    """
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # Create directories
    config.create_dirs()

    # Load Data
    print("Loading Data...")
    train_loader, val_loader, _ = get_dataloaders(config)

    # Initialize Model
    print("Initializing Model...")
    model = RNAModel(config).to(device)

    # Loss Function (MSE)
    criterion = nn.MSELoss()

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
    )

    # Training Loop
    best_score = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, config
        )

        # Validate
        val_score = validate(model, val_loader, device, config)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score}"
        )

        # Early Stopping & Model Saving
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"  >>> New Best Model Saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  >>> Patience: {patience_counter}/{config.PATIENCE}")

        if patience_counter >= config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation MCRMSE: {best_score}")


def generate_submission(config=Config):
    """
    Generates predictions for the test set and creates submission.csv.
    """
    set_seed(config.SEED)
    device = torch.device(config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders(config)

    # Load Model
    print("Loading best model for inference...")
    model = RNAModel(config).to(device)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    ids_list = []
    preds_list = []

    print("Generating predictions...")
    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            loop = batch["loop"].to(device)
            distance = batch["distance"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            # Output shape: (B, 107, 5)
            # We need predictions for ALL positions (0-106) for the submission file
            outputs = model(sequence, loop, distance)

            outputs_np = outputs.cpu().numpy()

            for i, sample_id in enumerate(batch_ids):
                # Shape: (107, 5)
                sample_pred = outputs_np[i]

                # We need to flatten this into rows: id_seqpos, val1, val2, ...
                # Create row identifiers
                for seqpos in range(config.SEQ_LEN):
                    row_id = f"{sample_id}_{seqpos}"
                    row_values = sample_pred[seqpos]

                    ids_list.append(row_id)
                    preds_list.append(row_values)

    # Create DataFrame
    cols = (
        config.TARGET_COLS
    )  # ['reactivity', 'deg_Mg_pH10', 'deg_pH10', 'deg_Mg_50C', 'deg_50C']
    df_preds = pd.DataFrame(preds_list, columns=cols)
    df_ids = pd.DataFrame({"id_seqpos": ids_list})

    submission_df = pd.concat([df_ids, df_preds], axis=1)

    # Save
    print(f"Saving submission to {config.SUBMISSION_PATH}...")
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run_experiment():
    """
    Runs the full pipeline: Training -> Inference.
    """
    train_model()
    generate_submission()

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.dataset import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Move batch to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)

        optimizer.zero_grad()

        # Forward pass: (Batch, Seq_Len, Num_Targets)
        preds = model(batch)
        targets = batch["targets"]

        # Slice predictions to match the scored sequence length (first 68 positions)
        # Targets are already shaped (Batch, 68, 3) in the dataset
        preds_scored = preds[:, : Config.SEQ_SCORED, :]

        # Compute Loss (MSE)
        loss = criterion(preds_scored, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimizer Step
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
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

            # Forward pass
            preds = model(batch)
            targets = batch["targets"]

            # Slice predictions to match scored length
            preds_scored = preds[:, : Config.SEQ_SCORED, :]

            # Accumulate
            all_preds.append(preds_scored.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE
    score = MCRMSE(all_targets, all_preds)
    return score.item()


def predict_and_submit(test_loader, device):
    """
    Generates predictions for the test set and saves the submission file.
    """
    print("Generating submission...")

    # Load Best Model
    model = RNAModel().to(device)
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print(f"Loaded best model from {Config.BEST_MODEL_PATH}")
    else:
        print("Warning: Best model checkpoint not found. Using current model state.")

    model.eval()
    results = []

    # Submission requires columns:
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    with torch.no_grad():
        for batch in test_loader:
            ids = batch["id"]
            # Move inputs to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Forward pass -> (Batch, 107, 3)
            # We predict for the full sequence length (107)
            preds = model(batch)
            preds = preds.cpu().numpy()

            # Iterate over samples in batch
            for i, sample_id in enumerate(ids):
                sample_preds = preds[i]  # Shape: (107, 3)

                # Iterate over all sequence positions
                for seqpos in range(Config.SEQ_LENGTH):
                    # Extract predicted values
                    # Model outputs: [reactivity, deg_Mg_pH10, deg_Mg_50C]
                    reactivity = sample_preds[seqpos, 0]
                    deg_Mg_pH10 = sample_preds[seqpos, 1]
                    deg_Mg_50C = sample_preds[seqpos, 2]

                    # Fill non-predicted columns with 0.0
                    deg_pH10 = 0.0
                    deg_50C = 0.0

                    # Construct Row ID
                    row_id = f"{sample_id}_{seqpos}"

                    results.append(
                        [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
                    )

    # Create DataFrame
    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    df_sub = pd.DataFrame(results, columns=columns)

    # Save to CSV
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training():
    """
    Main driver function for training, validation, and submission.
    """
    # 1. Setup
    set_seed(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. DataLoaders
    train_loader, val_loader, test_loader = get_dataloaders(
        debug=Config.DEBUG, load_cached_data=True
    )

    # 3. Model
    model = RNAModel().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # 5. Loss Function (MSE)
    criterion = nn.MSELoss()

    # 6. Training Loop
    best_score = float("inf")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Checkpoint
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved with MCRMSE: {best_score}")

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # 7. Inference & Submission
    predict_and_submit(test_loader, device)

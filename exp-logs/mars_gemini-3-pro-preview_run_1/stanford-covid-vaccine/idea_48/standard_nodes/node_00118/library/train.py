import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import Config
from library.utils import seed_everything, mcrmse
from library.dataset import get_dataloaders
from library.model import TopologicalWideResBiLSTM


def train_fn(model, loader, optimizer, criterion, device, config):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in loader:
        # Move batch to device
        sequence = batch["sequence"].to(device)
        loop_type = batch["loop_type"].to(device)
        rwpe = batch["rwpe"].to(device)
        distance = batch["distance"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

        batch_size = sequence.size(0)

        optimizer.zero_grad()

        # Forward pass
        # Output shape: (B, 107, 3)
        preds = model(sequence, loop_type, rwpe, distance)

        # Slice predictions to match the scored target length (first 68 positions)
        preds_sliced = preds[:, : config.pred_len, :]

        # Calculate Loss (Masked MSE)
        loss = criterion(preds_sliced, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Critical for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def eval_fn(model, loader, device, config):
    """
    Evaluates the model on the validation set using MCRMSE.
    """
    model.eval()
    preds_list = []
    targets_list = []

    with torch.no_grad():
        for batch in loader:
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            rwpe = batch["rwpe"].to(device)
            distance = batch["distance"].to(device)
            targets = batch["targets"].to(device)

            preds = model(sequence, loop_type, rwpe, distance)

            # Slice for evaluation metric calculation
            preds_sliced = preds[:, : config.pred_len, :]

            preds_list.append(preds_sliced.cpu())
            targets_list.append(targets.cpu())

    # Concatenate all batches
    preds_all = torch.cat(preds_list, dim=0)
    targets_all = torch.cat(targets_list, dim=0)

    # Calculate MCRMSE
    score = mcrmse(targets_all, preds_all)
    return score


def inference_fn(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["id"]
            sequence = batch["sequence"].to(device)
            loop_type = batch["loop_type"].to(device)
            rwpe = batch["rwpe"].to(device)
            distance = batch["distance"].to(device)

            # Predict for full sequence length (107)
            preds = model(sequence, loop_type, rwpe, distance)

            ids_list.extend(ids)
            preds_list.append(preds.cpu().numpy())

    preds_all = np.concatenate(preds_list, axis=0)  # Shape: (N, 107, 3)
    return ids_list, preds_all


def run_training(debug=False, epochs=20, batch_size=32):
    """
    Main function to orchestrate training, evaluation, and submission generation.
    """
    # Initialize Configuration
    config = Config(debug=debug, epochs=epochs, batch_size=batch_size)

    # Set seeds for reproducibility
    seed_everything(config.seed)

    # Load DataLoaders
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        config, load_cached_data=True
    )

    # Initialize Model
    print("Initializing Model...")
    model = TopologicalWideResBiLSTM(config)
    model.to(config.device)

    # Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    criterion = nn.MSELoss()

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(config.working_dir, "best_model.pth")

    print(f"Starting training for {config.epochs} epochs on {config.device}...")

    for epoch in range(config.epochs):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, criterion, config.device, config
        )

        # Validate
        val_score = eval_fn(model, val_loader, config.device, config)

        # Step Scheduler
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.6f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # Generate Submission
    print("Generating submission with best model...")

    # Load best model weights
    model.load_state_dict(torch.load(best_model_path, map_location=config.device))

    # Run inference
    ids, preds = inference_fn(model, test_loader, config.device)

    # Format submission
    # We need to map the 3 predicted columns to the 5 required columns
    # Predicted: [reactivity, deg_Mg_pH10, deg_Mg_50C]
    # Required: [reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]

    submission_rows = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # Shape (107, 3)

        for seqpos in range(config.seq_len):
            row_id = f"{sample_id}_{seqpos}"

            # Extract predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill unscored columns with 0.0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_rows.append(
                [row_id, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    columns = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    submission_df = pd.DataFrame(submission_rows, columns=columns)

    submission_df.to_csv(config.submission_file, index=False)
    print(f"Submission saved to {config.submission_file}")

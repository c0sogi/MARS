import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.utils import set_seed, MCRMSELoss
from library.data import get_dataloaders
from library.layers import RNAModel


def train_epoch(model, loader, optimizer, criterion, device, max_grad_norm):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        neighbor_indices = batch["neighbor_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, neighbor_indices, pair_masks)

        # Slice to scored length (first 68 positions) for loss calculation
        # This ensures we don't train on the zero-padded tails
        preds_sliced = preds[:, : Config.SEQ_SCORED, :]
        targets_sliced = targets[:, : Config.SEQ_SCORED, :]

        loss = criterion(preds_sliced, targets_sliced)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    Metric: MCRMSE on 3 scored columns and first 68 positions.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            neighbor_indices = batch["neighbor_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            preds = model(inputs, neighbor_indices, pair_masks)

            # Slice to scored length (68)
            preds = preds[:, : Config.SEQ_SCORED, :]
            targets = targets[:, : Config.SEQ_SCORED, :]

            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    # Global aggregation
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Filter for scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    # Config.TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    scored_indices = [0, 1, 3]

    preds_scored = all_preds[:, :, scored_indices]
    targets_scored = all_targets[:, :, scored_indices]

    # Flatten for MCRMSE calculation
    # Reshape to (N_total_positions, N_scored_columns)
    preds_flat = preds_scored.reshape(-1, len(scored_indices))
    targets_flat = targets_scored.reshape(-1, len(scored_indices))

    # Calculate MCRMSE
    mse = torch.mean((preds_flat - targets_flat) ** 2, dim=0)
    rmse = torch.sqrt(mse)
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


def inference(model, loader, device):
    """
    Generates predictions for the test set.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            neighbor_indices = batch["neighbor_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["id"]

            # Predict on full length (107)
            preds = model(inputs, neighbor_indices, pair_masks)

            preds_list.append(preds.cpu().numpy())
            ids_list.extend(ids)

    return np.concatenate(preds_list, axis=0), ids_list


def generate_submission(preds, ids, output_path):
    """
    Formats predictions into the submission CSV format.
    """
    # preds: (N_samples, 107, 5)
    N, L, C = preds.shape

    # Flatten predictions
    preds_flat = preds.reshape(-1, C)

    # Generate id_seqpos column
    id_seqpos = []
    for sample_id in ids:
        for i in range(L):
            id_seqpos.append(f"{sample_id}_{i}")

    # Create DataFrame
    df = pd.DataFrame(preds_flat, columns=Config.TARGET_COLS)
    df.insert(0, "id_seqpos", id_seqpos)

    # Save
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False):
    """
    Main execution function.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    if debug:
        Config.DEBUG = True
        Config.EPOCHS = 2
        print("Running in DEBUG mode.")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    model = RNAModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN)
    criterion = MCRMSELoss()

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs...")

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, Config.MAX_GRAD_NORM
        )
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse:.10f} | "
            f"Time: {elapsed:.2f}s"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  New best model saved! ({val_mcrmse:.10f})")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Inference
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    print("Generating predictions on test set...")
    test_preds, test_ids = inference(model, test_loader, device)

    generate_submission(test_preds, test_ids, Config.SUBMISSION_PATH)

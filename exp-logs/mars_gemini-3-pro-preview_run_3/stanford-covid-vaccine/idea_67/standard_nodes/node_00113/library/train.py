import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.config import Config
from library.utils import set_seed, MCRMSE
from library.data import get_dataloaders
from library.model import DADBiGRUModel


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_masks = batch["pair_masks"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs, pair_indices, pair_masks)

        # Calculate loss on all targets
        loss = criterion(outputs, targets)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices, pair_masks)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Calculate MCRMSE on scored columns only (reactivity, deg_Mg_pH10, deg_Mg_50C)
    # The MCRMSE function handles slicing to PRED_LEN (68) internally
    score = MCRMSE(all_targets, all_preds, scored_only=True)

    return score


def inference(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for batch in loader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_masks = batch["pair_masks"].to(device)
            ids = batch["ids"]

            outputs = model(inputs, pair_indices, pair_masks)

            all_preds.append(outputs.cpu().numpy())
            all_ids.extend(ids)

    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Running on device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model
    print("Initializing model...")
    model = DADBiGRUModel().to(device)

    # 4. Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX, eta_min=Config.ETA_MIN
    )

    # Use MSE Loss for training (optimizing MSE effectively optimizes RMSE)
    criterion = nn.MSELoss()

    # 5. Training Loop
    best_score = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Printing full precision
            f"Time: {elapsed:.2f}s"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! Score: {best_score}")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Val MCRMSE: {best_score}")

    # 6. Inference
    print("Generating submission...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    # Predict
    test_preds, test_ids = inference(model, test_loader, device)

    # Prepare Submission DataFrame
    # test_preds shape: (N_samples, 107, 5)
    # We need to flatten this to (N_samples * 107, 6) including id_seqpos

    N, L, C = test_preds.shape

    # Flatten predictions
    flat_preds = test_preds.reshape(-1, C)

    # Create id_seqpos column
    id_seqpos_list = []
    for sample_id in test_ids:
        for i in range(L):
            id_seqpos_list.append(f"{sample_id}_{i}")

    submission_df = pd.DataFrame(
        flat_preds,
        columns=["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"],
    )
    submission_df.insert(0, "id_seqpos", id_seqpos_list)

    # Save
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print(f"Submission shape: {submission_df.shape}")


if __name__ == "__main__":
    run_training()

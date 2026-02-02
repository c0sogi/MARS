import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os

from library.config import Config, set_seed
from library.dataset import get_dataloaders
from library.model import HybridResNetBiGRU
from library.loss import MaskedHuberLoss
from library.utils import mcrmse_metric


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Training loop for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        # Move inputs to device
        sequence = batch["sequence"].to(device)
        structure = batch["structure"].to(device)
        loop_type = batch["predicted_loop_type"].to(device)
        targets = batch["targets"].to(device)
        mask = batch["mask"].to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(sequence, structure, loop_type)

        # Compute loss
        loss = criterion(preds, targets, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * sequence.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Validation loop. Computes Loss and MCRMSE.
    """
    model.eval()
    running_loss = 0.0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            sequence = batch["sequence"].to(device)
            structure = batch["structure"].to(device)
            loop_type = batch["predicted_loop_type"].to(device)
            targets = batch["targets"].to(device)
            mask = batch["mask"].to(device)

            preds = model(sequence, structure, loop_type)

            loss = criterion(preds, targets, mask)
            running_loss += loss.item() * sequence.size(0)

            # Collect for MCRMSE calculation
            # Move to CPU to save GPU memory
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(dataloader.dataset)

    # Concatenate all batches
    all_preds = np.concatenate(all_preds, axis=0)  # (N, 107, 5)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 107, 5)

    # Slice to scored length for metric calculation
    # The competition scores the first 68 bases
    scored_preds = all_preds[:, : Config.SCORED_LEN, :]
    scored_targets = all_targets[:, : Config.SCORED_LEN, :]

    val_mcrmse = mcrmse_metric(scored_targets, scored_preds)

    return epoch_loss, val_mcrmse


def generate_submission(model, test_loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    ids_list = []
    preds_list = []

    with torch.no_grad():
        for batch in test_loader:
            sequence = batch["sequence"].to(device)
            structure = batch["structure"].to(device)
            loop_type = batch["predicted_loop_type"].to(device)
            batch_ids = batch["ids"]

            # Forward pass
            preds = model(sequence, structure, loop_type)  # (B, 107, 5)

            preds_list.append(preds.cpu().numpy())
            ids_list.extend(batch_ids)

    # Concatenate predictions
    all_preds = np.concatenate(preds_list, axis=0)  # (N_test, 107, 5)

    # Prepare data for DataFrame
    # We need to flatten: N_test * 107 rows
    flat_preds = all_preds.reshape(-1, 5)

    # Generate id_seqpos keys
    flat_ids = []
    for sample_id in ids_list:
        for i in range(Config.SEQ_LEN):
            flat_ids.append(f"{sample_id}_{i}")

    # Create DataFrame
    submission_df = pd.DataFrame(flat_preds, columns=Config.TARGET_COLS)
    submission_df.insert(0, "id_seqpos", flat_ids)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=Config.DEBUG
):
    """
    Main function to run the training pipeline.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Using device: {device}")

    # 1. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True, batch_size=batch_size, debug=debug
    )

    # 2. Model Initialization
    model = HybridResNetBiGRU().to(device)

    # 3. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.T_MAX)

    # 4. Loss Function
    criterion = MaskedHuberLoss(delta=Config.HUBER_DELTA)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = Config.MODEL_SAVE_PATH
    patience_counter = 0

    print("Starting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_mcrmse = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Print metrics (Full precision)
        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss} | "
            f"Val Loss: {val_loss} | "
            f"Val MCRMSE: {val_mcrmse}"
        )

        # Checkpointing (Save based on Validation Loss as requested)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    # 6. Generate Submission with Best Model
    print("Loading best model for submission generation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

    return model

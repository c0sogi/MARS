import os
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from library.config import Config
from library.utils import seed_all, MaskedMCRMSELoss, GlobalRMSETracker
from library.data import get_dataloaders
from library.model import DensePartnerAwareNet


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Performs one epoch of training.
    """
    model.train()
    running_loss = 0.0

    for inputs, partner_indices, targets in loader:
        inputs = inputs.to(device)
        partner_indices = partner_indices.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs, partner_indices)

        # Compute loss (MaskedMCRMSELoss handles the column selection internally)
        loss = criterion(preds, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, device):
    """
    Evaluates the model on the validation set using GlobalRMSETracker.
    """
    model.eval()
    tracker = GlobalRMSETracker(scored_indices=Config.SCORED_INDICES)

    with torch.no_grad():
        for inputs, partner_indices, targets in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)
            targets = targets.to(device)

            preds = model(inputs, partner_indices)

            # Update tracker with batch results
            tracker.update(preds, targets)

    return tracker.compute()


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    all_preds = []
    all_ids = []

    # Retrieve IDs from the dataset directly
    # The loader returns batches, but we need to align preds with IDs.
    # Since shuffle=False for test loader, we can iterate sequentially.
    dataset_ids = loader.dataset.ids
    current_idx = 0

    with torch.no_grad():
        for inputs, partner_indices in loader:
            inputs = inputs.to(device)
            partner_indices = partner_indices.to(device)

            # Forward pass
            preds = model(inputs, partner_indices)
            preds = preds.cpu().numpy()  # (B, Seq_Len, 5)

            batch_size = preds.shape[0]
            batch_ids = dataset_ids[current_idx : current_idx + batch_size]
            current_idx += batch_size

            all_preds.append(preds)
            all_ids.extend(batch_ids)

    all_preds = np.concatenate(all_preds, axis=0)  # (N_samples, Seq_Len, 5)

    # Prepare submission data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = Config.ALL_TARGET_COLS

    for i, sample_id in enumerate(all_ids):
        sample_preds = all_preds[i]  # (107, 5)
        for seqpos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seqpos}"
            row_values = sample_preds[seqpos].tolist()

            # Create dictionary for DataFrame
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    # Create DataFrame and save
    submission_df = pd.DataFrame(submission_rows)
    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug=False):
    """
    Main function to run the training pipeline.
    """
    seed_all(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 2. Initialize Model
    model = DensePartnerAwareNet(
        input_channels=Config.INPUT_CHANNELS,
        tcn_channels=Config.TCN_CHANNELS,
        tcn_layers=Config.TCN_LAYERS,
        kernel_size=Config.TCN_KERNEL_SIZE,
        dropout=Config.DROPOUT,
        latent_dim=Config.LATENT_DIM,
        gru_hidden=Config.GRU_HIDDEN_DIM,
        num_targets=Config.NUM_TARGETS,
    ).to(device)

    # 3. Setup Optimization
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    criterion = MaskedMCRMSELoss(scored_indices=Config.SCORED_INDICES)

    # 4. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_mcrmse = evaluate(model, val_loader, device)

        # Scheduler step
        scheduler.step(val_mcrmse)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Early Stopping and Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"  New best model saved! MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    # 5. Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

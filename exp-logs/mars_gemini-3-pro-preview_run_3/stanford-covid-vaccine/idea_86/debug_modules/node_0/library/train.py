import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, calculate_mcrmse
from library.loss import MCRMSELoss
from library.data import get_dataloaders
from library.model import RNAModel


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training DataLoader.
        criterion: Loss function (MCRMSELoss).
        optimizer: Optimizer.
        device: Torch device.
        epoch: Current epoch number.

    Returns:
        avg_loss: Average training loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        inputs = batch["inputs"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        pair_mask = batch["pair_mask"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs, pair_indices, pair_mask)

        # Calculate loss
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory for stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)

        # Optimizer step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions before calculating MCRMSE.

    Args:
        model: The PyTorch model.
        dataloader: Validation DataLoader.
        device: Torch device.

    Returns:
        mcrmse: The calculated MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(inputs, pair_indices, pair_mask)

            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    if not all_preds:
        return 0.0

    full_preds = torch.cat(all_preds, dim=0)
    full_targets = torch.cat(all_targets, dim=0)

    # Calculate metric using the utility function
    mcrmse = calculate_mcrmse(full_preds, full_targets)

    return mcrmse


def generate_submission(model, dataloader, device, submission_path):
    """
    Generates predictions for the test set and saves the submission file.

    Args:
        model: The trained PyTorch model.
        dataloader: Test DataLoader.
        device: Torch device.
        submission_path: Path to save the CSV.
    """
    model.eval()
    results = []

    # Columns required in submission
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["inputs"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            pair_mask = batch["pair_mask"].to(device)
            ids = batch["id"]  # List of IDs

            outputs = model(inputs, pair_indices, pair_mask)
            outputs = outputs.cpu().numpy()  # (B, 107, 5)

            # Iterate through batch
            for i, sample_id in enumerate(ids):
                sample_preds = outputs[i]  # (107, 5)
                seq_len = sample_preds.shape[0]

                for seq_pos in range(seq_len):
                    row_id = f"{sample_id}_{seq_pos}"
                    row_data = {
                        "id_seqpos": row_id,
                        "reactivity": sample_preds[seq_pos, 0],
                        "deg_Mg_pH10": sample_preds[seq_pos, 1],
                        "deg_pH10": sample_preds[seq_pos, 2],
                        "deg_Mg_50C": sample_preds[seq_pos, 3],
                        "deg_50C": sample_preds[seq_pos, 4],
                    }
                    results.append(row_data)

    # Create DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure column order
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    # Save
    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_training(debug=False, debug_size=100):
    """
    Main execution function for training, validation, and submission generation.
    """
    # 1. Setup
    Config.setup()
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=True,
        debug=debug,
        debug_size=debug_size,
        batch_size=Config.BATCH_SIZE,
    )

    # 3. Model
    model = RNAModel().to(device)

    # 4. Optimizer & Loss
    criterion = MCRMSELoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler: Cosine Annealing
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    # 5. Training Loop
    best_mcrmse = float("inf")
    patience_counter = 0

    print("Starting training...")

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Step Scheduler
        scheduler.step()

        # Logging (Full Precision)
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_mcrmse}"
        )

        # Early Stopping & Checkpointing
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with MCRMSE: {best_mcrmse}")
        else:
            patience_counter += 1
            print(
                f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Generate Submission
    print("Loading best model for submission...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

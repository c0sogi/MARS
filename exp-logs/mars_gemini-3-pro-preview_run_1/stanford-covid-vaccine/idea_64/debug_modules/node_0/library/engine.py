import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import time

from library.config import Config
from library.utils import set_seed, mcrmse_metric, save_checkpoint
from library.data import get_dataloaders
from library.model import StructureInjectedWideResBiLSTM


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    """
    Trains the model for one epoch.

    Args:
        model: The PyTorch model.
        dataloader: Training dataloader.
        optimizer: The optimizer.
        device: 'cuda' or 'cpu'.
        epoch: Current epoch number (for logging).

    Returns:
        avg_loss: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    # Loss function: MSE
    criterion = nn.MSELoss()

    for batch in dataloader:
        # Move data to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["dist"].to(device)
        targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

        # Forward pass
        # Model outputs (B, 107, 3)
        outputs = model(seq, loop, dist)

        # Masked Loss Calculation
        # We only calculate loss on the first PRED_LEN (68) positions
        # Slice outputs to match targets shape
        outputs_scored = outputs[:, : Config.PRED_LEN, :]

        loss = criterion(outputs_scored, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    avg_loss = running_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set.

    Args:
        model: The PyTorch model.
        dataloader: Validation dataloader.
        device: 'cuda' or 'cpu'.

    Returns:
        mcrmse: The MCRMSE score.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            targets = batch["targets"].to(device)  # Shape: (B, 68, 3)

            # Forward pass
            outputs = model(seq, loop, dist)

            # Slice to scored positions
            outputs_scored = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_scored.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Concatenate all batches
    # Shape: (Total_Samples, 68, 3)
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)

    # Calculate metric
    score = mcrmse_metric(y_true, y_pred)

    return score


def generate_submission(model_path, device):
    """
    Generates the submission file using the trained model.

    Args:
        model_path: Path to the saved best model checkpoint.
        device: 'cuda' or 'cpu'.
    """
    # 1. Load Model
    model = StructureInjectedWideResBiLSTM()
    model.to(device)

    # Load checkpoint
    if not os.path.exists(model_path):
        print(
            f"Model path {model_path} does not exist. Skipping submission generation."
        )
        return

    checkpoint = torch.load(model_path, map_location=device)
    # Handle case where checkpoint saves 'state_dict' vs direct model state
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    # 2. Load Test Data
    # shuffle=False is critical for submission alignment
    test_loader = get_dataloaders(
        split="test", batch_size=Config.BATCH_SIZE, shuffle=False
    )

    # 3. Inference
    ids_list = []
    preds_list = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["dist"].to(device)
            ids = batch["id"]

            # Forward
            outputs = model(seq, loop, dist)  # (B, 107, 3)

            # Move to CPU
            outputs = outputs.cpu().numpy()

            ids_list.extend(ids)
            preds_list.append(outputs)

    # Concatenate predictions: (N_test, 107, 3)
    all_preds = np.concatenate(preds_list, axis=0)

    # 4. Format Submission
    # We need to flatten this to (N_test * 107) rows
    # Columns required: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    # Model predicts: reactivity (idx 0), deg_Mg_pH10 (idx 1), deg_Mg_50C (idx 2)
    # Missing: deg_pH10, deg_50C (fill with 0)

    submission_data = []

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # Shape (107, 3)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID
            id_seqpos = f"{sample_id}_{seqpos}"

            # Predictions
            reactivity = sample_preds[seqpos, 0]
            deg_Mg_pH10 = sample_preds[seqpos, 1]
            deg_Mg_50C = sample_preds[seqpos, 2]

            # Fill others with 0
            deg_pH10 = 0.0
            deg_50C = 0.0

            submission_data.append(
                [id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C]
            )

    # Create DataFrame
    cols = [
        "id_seqpos",
        "reactivity",
        "deg_Mg_pH10",
        "deg_pH10",
        "deg_Mg_50C",
        "deg_50C",
    ]
    sub_df = pd.DataFrame(submission_data, columns=cols)

    # Save
    os.makedirs("./submission", exist_ok=True)
    out_path = "./submission/submission.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")


def train_model(max_epochs=Config.EPOCHS, max_samples=None):
    """
    Main function to train the model.

    Args:
        max_epochs: Number of epochs to train.
        max_samples: Limit dataset size for debugging.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data Loaders
    train_loader = get_dataloaders(
        split="train", batch_size=Config.BATCH_SIZE, max_samples=max_samples
    )
    val_loader = get_dataloaders(
        split="val", batch_size=Config.BATCH_SIZE, max_samples=max_samples
    )

    # 2. Model
    model = StructureInjectedWideResBiLSTM()
    model.to(device)

    # 3. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.T_MAX
    )

    # 4. Training Loop
    best_mcrmse = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print(f"Starting training for {max_epochs} epochs on {device}...")

    for epoch in range(1, max_epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_mcrmse = validate(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Logging
        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch}/{max_epochs} | "
            f"Time: {elapsed:.1f}s | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_mcrmse}"
        )  # Full precision print

        # Checkpoint
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            save_checkpoint(model.state_dict(), best_model_path)
            print(f"  >>> New Best Model Saved (MCRMSE: {best_mcrmse})")

    print("Training complete.")

    # 5. Generate Submission
    generate_submission(best_model_path, device)

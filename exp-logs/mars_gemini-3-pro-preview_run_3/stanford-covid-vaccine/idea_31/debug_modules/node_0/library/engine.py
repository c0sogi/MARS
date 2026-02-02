import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import os
from library.config import Config
from library.dataset import RNADataset
from library.model import SDCGBiGRU
from library.loss import MCRMSELoss


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_fn(model, dataloader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    # Gradient clipping value from config
    clip_norm = Config.GRAD_CLIP_NORM

    for batch in dataloader:
        inputs = batch["input"].to(device)
        pair_indices = batch["pair_indices"].to(device)
        targets = batch["target"].to(device)  # Shape: (B, 68, 5)

        optimizer.zero_grad()

        # Forward pass
        # Model output shape: (B, 107, 5)
        outputs = model(inputs, pair_indices)

        # Slice outputs to match targets (first 68 positions)
        outputs_sliced = outputs[:, : Config.PRED_LEN, :]

        # Compute loss on all 5 columns as per strategy
        loss = criterion(outputs_sliced, targets)

        loss.backward()

        # Explicit Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

        optimizer.step()

        # Accumulate loss (weighted by batch size for correct average)
        running_loss += loss.item() * inputs.size(0)
        count += inputs.size(0)

    return running_loss / count


def eval_fn(model, dataloader, device):
    """
    Evaluates the model on the validation set using the competition metric.
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = batch["input"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            targets = batch["target"].to(device)  # Shape: (B, 68, 5)

            outputs = model(inputs, pair_indices)

            # Slice outputs to scored length
            outputs_sliced = outputs[:, : Config.PRED_LEN, :]

            all_preds.append(outputs_sliced.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches to avoid batch-size bias in metric
    all_preds = torch.cat(all_preds, dim=0)  # (N, 68, 5)
    all_targets = torch.cat(all_targets, dim=0)  # (N, 68, 5)

    # Identify indices of scored columns
    # TARGET_COLS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # SCORED_COLS = ["reactivity", "deg_Mg_pH10", "deg_Mg_50C"]
    target_cols = Config.TARGET_COLS
    scored_cols = Config.SCORED_COLS
    scored_indices = [i for i, col in enumerate(target_cols) if col in scored_cols]

    # Filter predictions and targets to only the 3 scored columns
    filtered_preds = all_preds[:, :, scored_indices]
    filtered_targets = all_targets[:, :, scored_indices]

    # Compute MCRMSE manually
    # Mean over samples (dim 0) and sequence length (dim 1) -> MSE per column
    mse = torch.mean((filtered_preds - filtered_targets) ** 2, dim=(0, 1))
    # Sqrt -> RMSE per column
    rmse = torch.sqrt(mse)
    # Mean -> MCRMSE
    mcrmse = torch.mean(rmse)

    return mcrmse.item()


def run_training():
    """
    Main training loop with Early Stopping.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print(f"Starting training on device: {device}")

    # Initialize Datasets and Loaders
    train_dataset = RNADataset(split="train")
    val_dataset = RNADataset(split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Initialize Model
    model = SDCGBiGRU().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # Loss Function
    criterion = MCRMSELoss()

    # Training State
    best_score = float("inf")
    patience_counter = 0

    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(model, train_loader, optimizer, criterion, device)

        # Validate
        val_score = eval_fn(model, val_loader, device)

        # Update Scheduler
        scheduler.step()

        # Logging
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val MCRMSE: {val_score}"
        )

        # Early Stopping Check
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

    print(f"Training complete. Best Val Score: {best_score}")


def generate_submission():
    """
    Generates submission file for the test set.
    """
    set_seed(Config.SEED)
    device = Config.DEVICE
    print("Generating submission...")

    # Load Test Data
    test_dataset = RNADataset(split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Best Model
    model = SDCGBiGRU().to(device)
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    else:
        print(
            f"Warning: Model file {Config.MODEL_SAVE_PATH} not found. Using untrained model."
        )

    model.eval()

    preds_map = {}

    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input"].to(device)
            pair_indices = batch["pair_indices"].to(device)
            ids = batch["id"]

            # Forward pass (Full length 107)
            outputs = model(inputs, pair_indices)  # (B, 107, 5)
            outputs = outputs.cpu().numpy()

            for i, sample_id in enumerate(ids):
                preds_map[sample_id] = outputs[i]

    # Format Submission
    submission_data = []
    target_cols = Config.TARGET_COLS

    # Iterate through test dataset IDs to maintain order
    for sample_id in test_dataset.ids:
        # Get prediction matrix (107, 5)
        pred_matrix = preds_map[sample_id]

        # Create row for each sequence position
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_preds = pred_matrix[seqpos]

            row_dict = {"id_seqpos": row_id}
            for idx, col in enumerate(target_cols):
                row_dict[col] = float(row_preds[idx])

            submission_data.append(row_dict)

    submission_df = pd.DataFrame(submission_data)

    # Save to CSV
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

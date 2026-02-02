import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import RNADataset
from library.model import DeepStabilizedBiGRU
from library.metrics import MCRMSELoss, scored_mcrmse


def set_seed(seed):
    """
    Sets the random seed for reproducibility across Python, NumPy, and PyTorch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    Performs one epoch of training.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the training set.
        optimizer (Optimizer): The optimizer.
        criterion (nn.Module): The loss function.
        device (torch.device): The computing device (CPU or GPU).

    Returns:
        float: The average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        features = batch["features"].to(device)
        adjacency = batch["adjacency"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(features, adjacency)

        # Compute loss
        # MCRMSELoss handles slicing internally
        loss = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient Clipping (Mandatory max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        # Optimization step
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / num_batches if num_batches > 0 else 0.0


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    Aggregates predictions globally to compute accurate MCRMSE metrics.

    Args:
        model (nn.Module): The neural network model.
        dataloader (DataLoader): DataLoader for the validation set.
        criterion (nn.Module): The loss function.
        device (torch.device): The computing device.

    Returns:
        tuple: (average_loss, scored_metric_value)
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in dataloader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            targets = batch["targets"].to(device)

            outputs = model(features, adjacency)

            # Store predictions and targets for global metric calculation
            all_preds.append(outputs.cpu())
            all_targets.append(targets.cpu())

    # Concatenate all batches
    if not all_preds:
        return 0.0, 0.0

    global_preds = torch.cat(all_preds, dim=0)
    global_targets = torch.cat(all_targets, dim=0)

    # Compute Loss (MCRMSE on all 5 targets)
    # We use the criterion class logic but on the full tensors
    # Note: criterion expects inputs on device usually, but here we are on CPU.
    # MCRMSELoss uses torch operations which work on CPU tensors.
    val_loss = criterion(global_preds, global_targets).item()

    # Compute Scored Metric (MCRMSE on 3 specific columns, sliced to 68)
    val_score = scored_mcrmse(global_preds, global_targets)

    return val_loss, val_score


def train_and_evaluate(load_cached_data=True, max_samples=None):
    """
    Main driver function to train the model with Early Stopping and save the best checkpoint.

    Args:
        load_cached_data (bool): Whether to load pre-processed data from cache.
        max_samples (int, optional): Limit dataset size for debugging.
    """
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Initializing Datasets...")
    train_dataset = RNADataset(
        split="train", load_cached_data=load_cached_data, max_samples=max_samples
    )
    val_dataset = RNADataset(
        split="val", load_cached_data=load_cached_data, max_samples=max_samples
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing Model...")
    model = DeepStabilizedBiGRU().to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )
    criterion = MCRMSELoss()

    # 5. Training Loop with Early Stopping
    best_score = float("inf")
    patience_counter = 0

    print("Starting Training...")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_loss, val_score = validate(model, val_loader, criterion, device)

        # Scheduler Step
        scheduler.step()

        # Logging (Full precision)
        print(
            f"Epoch {epoch + 1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score (MCRMSE): {val_score}"
        )

        # Early Stopping Logic
        # We optimize for the competition metric (Val Score)
        if val_score < best_score:
            best_score = val_score
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with score: {best_score}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break

    print(f"Training complete. Best Val Score: {best_score}")


def generate_submission(load_cached_data=True):
    """
    Loads the best model, performs inference on the test set, and saves the submission file.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Test Data
    test_dataset = RNADataset(split="test", load_cached_data=load_cached_data)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load Model
    model = DeepStabilizedBiGRU().to(device)
    if os.path.exists(Config.MODEL_PATH):
        model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_PATH}")
    else:
        print("Warning: No trained model found. Using random initialization.")

    model.eval()

    # Inference
    ids = []
    preds = []

    print("Running Inference on Test Set...")
    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            adjacency = batch["adjacency"].to(device)
            batch_ids = batch["id"]

            # Forward pass
            outputs = model(features, adjacency)

            # Move to CPU
            outputs = outputs.cpu().numpy()

            ids.extend(batch_ids)
            preds.append(outputs)

    # Concatenate predictions: (N_samples, 107, 5)
    preds = np.concatenate(preds, axis=0)

    # Prepare Submission Data
    # Format: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)

        for seqpos in range(Config.SEQ_LEN):
            # Row ID
            row_id = f"{sample_id}_{seqpos}"

            # Values
            row_values = sample_preds[seqpos].tolist()

            # Construct row dict
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = row_values[col_idx]

            submission_rows.append(row_dict)

    # Create DataFrame and Save
    import pandas as pd

    submission_df = pd.DataFrame(submission_rows)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + target_cols
    submission_df = submission_df[cols]

    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

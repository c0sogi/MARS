import os
import torch
import numpy as np
import pandas as pd
import time
from library.config import Config
from library.model import HC_SDRN
from library.loss import MaskedMCRMSELoss
from library.data import get_dataloaders
from library.utils import set_seed, compute_global_mcrmse


def train_one_epoch(model, loader, optimizer, criterion, device, debug=False):
    """
    Runs one epoch of training.

    Args:
        model: The HC_SDRN model.
        loader: DataLoader for training data.
        optimizer: Optimizer instance.
        criterion: Loss function instance.
        device: Torch device.
        debug (bool): If True, limits to a few batches for debugging.

    Returns:
        float: Average loss for the epoch.
    """
    model.train()
    running_loss = 0.0
    count = 0

    for i, (x, p_idx, y) in enumerate(loader):
        x = x.to(device)
        p_idx = p_idx.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # Forward pass returns tuple (y1, y2)
        y1, y2 = model(x, p_idx)

        # Calculate loss for both passes
        loss1 = criterion(y1, y)
        loss2 = criterion(y2, y)

        # Composite loss: weighted sum
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        count += 1

        if debug and i >= 5:
            break

    return running_loss / count


def train_model(epochs=Config.EPOCHS, debug=False, save_dir=Config.CACHE_DIR):
    """
    Main training routine.

    Args:
        epochs (int): Number of epochs to train.
        debug (bool): If True, runs a shorter debug cycle.
        save_dir (str): Directory to save the best model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get DataLoaders
    train_loader, val_loader, _, _ = get_dataloaders(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Initialize Model, Optimizer, Scheduler, Loss
    model = HC_SDRN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )
    criterion = MaskedMCRMSELoss()

    best_score = float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        start_time = time.time()

        # Training Step
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, debug
        )

        # Validation Step
        # compute_global_mcrmse handles the evaluation loop and metric calculation
        val_score = compute_global_mcrmse(model, val_loader, device)

        elapsed = time.time() - start_time

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {elapsed:.2f}s | "
            f"Train Loss: {train_loss:.10f} | Val MCRMSE: {val_score:.10f}"
        )

        # Scheduler Step
        scheduler.step(val_score)

        # Save Best Model
        if val_score < best_score:
            print(
                f"Validation score improved from {best_score:.10f} to {val_score:.10f}. Saving model..."
            )
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Validation Score: {best_score:.10f}")
    return best_model_path


def generate_submission(model_path, output_path=Config.SUBMISSION_PATH, debug=False):
    """
    Generates submission file for the test set.

    Args:
        model_path (str): Path to the saved model weights.
        output_path (str): Path to save the submission CSV.
        debug (bool): If True, limits inference size.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Data
    _, _, test_loader, test_ids = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        train_shuffle=False,
    )

    # Load Model
    model = HC_SDRN().to(device)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        raise FileNotFoundError(f"Model file not found at {model_path}")

    model.eval()
    all_preds = []

    print("Generating predictions...")
    with torch.no_grad():
        for i, (x, p_idx) in enumerate(test_loader):
            x = x.to(device)
            p_idx = p_idx.to(device)

            # Forward pass returns (y1, y2). We use y2 (refined).
            _, y2 = model(x, p_idx)

            # Move to CPU and numpy
            preds = y2.cpu().numpy()  # Shape: (Batch, Seq_Len, 5)
            all_preds.append(preds)

            if debug and i >= 5:
                break

    # Concatenate all batches
    # Shape: (N_Samples, Seq_Len, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare Submission Data
    # We need to flatten: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []

    # Ensure we iterate only up to the number of samples we have predictions for
    # (relevant if debug=True)
    num_samples = len(all_preds)
    current_ids = test_ids[:num_samples]

    for i, sample_id in enumerate(current_ids):
        # For each position in the sequence
        for pos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{pos}"
            row_preds = all_preds[i, pos]  # Array of 5 values

            # Create row dict or list
            row_data = [row_id] + row_preds.tolist()
            submission_rows.append(row_data)

    # Create DataFrame
    cols = ["id_seqpos"] + Config.TARGET_COLS
    sub_df = pd.DataFrame(submission_rows, columns=cols)

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

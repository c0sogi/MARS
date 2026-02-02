import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from library.config import Config
from library.data import get_dataloaders
from library.model import RHIGFN
from library.loss import mcrmse_loss, GlobalMCRMSE


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device, epoch):
    """
    Executes one training epoch with the Iterative Refinement Loop.
    """
    model.train()
    running_loss = 0.0

    # Progress bar for monitoring
    # pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]", leave=False)

    for batch_idx, (x, p_idx, y) in enumerate(loader):
        x = x.to(device)
        p_idx = p_idx.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # =====================================================================
        # Iterative Refinement Loop
        # =====================================================================

        # --- Pass 1: Static (Zero Feedback) ---
        # Forward pass with no feedback
        pred_1 = model(x, p_idx, feedback=None)

        # Calculate Aux Loss (Strict masking is handled inside mcrmse_loss)
        loss_1 = mcrmse_loss(pred_1, y)

        # --- Pass 2: Refinement (With Feedback) ---
        # Detach gradients from Pass 1 to stop backprop through time/iterations
        # The FeedbackModule inside the model handles channel masking.
        feedback = pred_1.detach()

        pred_2 = model(x, p_idx, feedback=feedback)

        # Calculate Primary Loss
        loss_2 = mcrmse_loss(pred_2, y)

        # --- Total Loss ---
        # Weighted sum
        total_loss = loss_2 + (Config.AUX_LOSS_WEIGHT * loss_1)

        # Optimization
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
        optimizer.step()

        running_loss += total_loss.item()

    avg_loss = running_loss / len(loader)
    return avg_loss


def validate(model, loader, device):
    """
    Validates the model using Global MCRMSE.
    """
    model.eval()
    metric = GlobalMCRMSE()

    with torch.no_grad():
        for x, p_idx, y in loader:
            x = x.to(device)
            p_idx = p_idx.to(device)
            y = y.to(device)

            # --- Pass 1 ---
            pred_1 = model(x, p_idx, feedback=None)

            # --- Pass 2 ---
            feedback = (
                pred_1  # No need to detach in no_grad mode, but conceptually same
            )
            pred_2 = model(x, p_idx, feedback=feedback)

            # Update Global Metric with final predictions
            metric.update(pred_2, y)

    return metric.compute()


def generate_submission(model, loader, test_ids, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    print("Generating submission...")
    model.eval()

    all_preds = []

    with torch.no_grad():
        for x, p_idx, _ in loader:  # Test loader yields dummy targets
            x = x.to(device)
            p_idx = p_idx.to(device)

            # --- Pass 1 ---
            pred_1 = model(x, p_idx, feedback=None)

            # --- Pass 2 ---
            pred_2 = model(x, p_idx, feedback=pred_1)

            # Move to CPU and numpy
            preds_np = pred_2.cpu().numpy()  # (B, 107, 5)
            all_preds.append(preds_np)

    # Concatenate all batches
    # Shape: (N_samples, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # Prepare submission data
    submission_rows = []
    target_cols = (
        Config.TARGET_COLS
    )  # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(test_ids):
        sample_preds = all_preds[i]  # (107, 5)

        for seq_pos in range(Config.SEQ_LENGTH):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_preds[seq_pos]

            # Create dictionary for row
            row_dict = {"id_seqpos": row_id}
            for col_idx, col_name in enumerate(target_cols):
                row_dict[col_name] = float(row_values[col_idx])

            submission_rows.append(row_dict)

    # Create DataFrame
    df_sub = pd.DataFrame(submission_rows)

    # Save
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders(
        debug=Config.DEBUG
    )

    # 3. Model
    print("Initializing model...")
    model = RHIGFN().to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=Config.SCHEDULER_FACTOR,
        patience=Config.SCHEDULER_PATIENCE,
        verbose=True,
    )

    # 5. Training Loop
    best_score = float("inf")
    early_stop_counter = 0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_score = validate(model, val_loader, device)

        # Scheduler Step
        scheduler.step(val_score)

        # Logging
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val MCRMSE: {val_score} | "  # Full precision as requested
            f"LR: {current_lr:.2e}"
        )

        # Checkpointing & Early Stopping
        if val_score < best_score:
            best_score = val_score
            early_stop_counter = 0
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"  >>> New Best Model Saved! Score: {best_score}")
        else:
            early_stop_counter += 1
            print(
                f"  Early Stopping Counter: {early_stop_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        if early_stop_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # 6. Submission
    print("\nTraining complete. Generating submission with best model...")

    # Load best model
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))

    generate_submission(model, test_loader, test_ids, device, Config.SUBMISSION_PATH)


if __name__ == "__main__":
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    run_training()

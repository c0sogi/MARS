import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim

from library.config import (
    WORKING_DIR,
    MODEL_PATH,
    SUBMISSION_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    SEQ_SCORED,
    SEQ_LEN,
    SUBMISSION_COLS,
    SEED,
)
from library.utils import seed_everything, calculate_mcrmse
from library.data import get_dataset
from library.model import RNANet, HomoscedasticLoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one training epoch.
    """
    model.train()
    total_loss = 0.0
    total_mse_val = 0.0
    total_mse_unc = 0.0

    for batch in loader:
        # Move inputs to device
        seq = batch["seq"].to(device)
        loop = batch["loop"].to(device)
        dist = batch["pair_dist"].to(device)
        target = batch["target"].to(device)
        error = batch["error"].to(device)

        optimizer.zero_grad()

        # Forward pass
        pred_val, pred_unc = model(seq, loop, dist)

        # Slice to scored positions (first 68) for loss calculation
        pred_val_scored = pred_val[:, :SEQ_SCORED, :]
        pred_unc_scored = pred_unc[:, :SEQ_SCORED, :]

        # Compute Loss
        loss, mse_val, mse_unc = criterion(
            pred_val_scored, target, pred_unc_scored, error
        )

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mse_val += mse_val.item()
        total_mse_unc += mse_unc.item()

    avg_loss = total_loss / len(loader)
    avg_mse_val = total_mse_val / len(loader)
    avg_mse_unc = total_mse_unc / len(loader)

    return avg_loss, avg_mse_val, avg_mse_unc


def validate(model, loader, device):
    """
    Evaluates the model on the validation set.
    Returns the MCRMSE score.
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["pair_dist"].to(device)
            target = batch["target"].cpu().numpy()

            # Forward pass
            pred_val, _ = model(seq, loop, dist)

            # Slice to scored positions (first 68)
            pred_val_scored = pred_val[:, :SEQ_SCORED, :].cpu().numpy()

            all_preds.append(pred_val_scored)
            all_targets.append(target)

    # Concatenate all batches
    y_pred = np.concatenate(all_preds, axis=0)  # [N, 68, 3]
    y_true = np.concatenate(all_targets, axis=0)  # [N, 68, 3]

    # Flatten for MCRMSE calculation: [N*68, 3]
    # Note: calculate_mcrmse expects [N, 3], so we flatten the sequence dimension into the batch dimension
    y_pred_flat = y_pred.reshape(-1, 3)
    y_true_flat = y_true.reshape(-1, 3)

    score = calculate_mcrmse(y_true_flat, y_pred_flat)
    return score


def predict(model, loader, device):
    """
    Generates predictions for the test set.
    Returns an array of shape [N_samples, 107, 5] formatted for submission.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            loop = batch["loop"].to(device)
            dist = batch["pair_dist"].to(device)

            # Forward pass: Get predictions for full sequence length (107)
            pred_val, _ = model(seq, loop, dist)  # [B, 107, 3]
            pred_val = pred_val.cpu().numpy()

            batch_size = pred_val.shape[0]

            # Initialize full prediction array [B, 107, 5]
            # Columns: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            full_preds = np.zeros((batch_size, SEQ_LEN, 5), dtype=np.float32)

            # Map model outputs (3 cols) to submission format (5 cols)
            # Model outputs: 0:reactivity, 1:deg_Mg_pH10, 2:deg_Mg_50C
            # Submission:    0:reactivity, 1:deg_Mg_pH10, 2:deg_pH10, 3:deg_Mg_50C, 4:deg_50C

            full_preds[:, :, 0] = pred_val[:, :, 0]  # reactivity
            full_preds[:, :, 1] = pred_val[:, :, 1]  # deg_Mg_pH10
            full_preds[:, :, 3] = pred_val[:, :, 2]  # deg_Mg_50C
            # deg_pH10 (idx 2) and deg_50C (idx 4) remain 0.0

            all_preds.append(full_preds)

    return np.concatenate(all_preds, axis=0)


def run_training():
    """
    Main pipeline to train the model and generate submission.
    """
    # 1. Setup
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

    # 2. Data Loading
    print("Loading datasets...")
    train_ds = get_dataset("train", load_cached_data=True)
    val_ds = get_dataset("val", load_cached_data=True)
    test_ds = get_dataset("test", load_cached_data=True)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
    )

    # 3. Model Initialization
    model = RNANet().to(device)
    criterion = HomoscedasticLoss().to(device)

    # Optimizer: Parameters of both model and learnable loss weights
    optimizer = optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # 4. Training Loop
    best_mcrmse = float("inf")

    print(f"Starting training for {EPOCHS} epochs...")
    for epoch in range(EPOCHS):
        train_loss, train_mse_val, train_mse_unc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_mcrmse = validate(model, val_loader, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1:02d} | "
            f"Loss: {train_loss:.8f} | "
            f"Train MSE(Val): {train_mse_val:.8f} | "
            f"Train MSE(Unc): {train_mse_unc:.8f} | "
            f"Val MCRMSE: {val_mcrmse:.12f}"
        )

        # Save best model
        if val_mcrmse < best_mcrmse:
            best_mcrmse = val_mcrmse
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  >>> New Best Model Saved! (Previous: {best_mcrmse:.12f})")

    print(f"Training complete. Best Validation MCRMSE: {best_mcrmse:.12f}")

    # 5. Inference and Submission
    print("Generating submission...")

    # Load best model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    else:
        print("Warning: Best model not found. Using current model state.")

    predictions = predict(model, test_loader, device)  # [N_samples, 107, 5]

    # Format submission
    # We need to iterate over samples and sequence positions to create the id_seqpos column
    submission_rows = []
    test_ids = test_ds.ids

    for i, sample_id in enumerate(test_ids):
        sample_pred = predictions[i]  # [107, 5]
        for seq_pos in range(SEQ_LEN):
            row_id = f"{sample_id}_{seq_pos}"
            row_values = sample_pred[seq_pos].tolist()
            submission_rows.append([row_id] + row_values)

    submission_df = pd.DataFrame(
        submission_rows, columns=["id_seqpos"] + SUBMISSION_COLS
    )
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    run_training()

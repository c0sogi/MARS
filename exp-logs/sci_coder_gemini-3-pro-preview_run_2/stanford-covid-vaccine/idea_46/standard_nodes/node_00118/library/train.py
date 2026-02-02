import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from library.config import Config
from library.data import process_data, RNADataset
from library.model import SDFRNModel
from library.loss import mcrmse_loss


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in loader:
        seq = batch["seq"].to(device)
        struct = batch["struct"].to(device)
        loop = batch["loop"].to(device)
        pid = batch["partner_id"].to(device)
        pidx = batch["partner_idx"].to(device)
        targets = batch["targets"].to(device)

        optimizer.zero_grad()

        # Pass 1: Initial prediction with zero feedback
        pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)

        # Pass 2: Refined prediction using detached Pass 1 output as feedback
        pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1.detach())

        # Calculate loss
        loss1 = mcrmse_loss(pred1, targets)
        loss2 = mcrmse_loss(pred2, targets)
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()

    # Accumulators for Global RMSE
    # We score 3 columns: reactivity, deg_Mg_pH10, deg_Mg_50C
    sum_squared_errors = torch.zeros(3, device=device)
    total_scored_positions = 0

    scored_cols_indices = Config.TARGET_INDICES  # [0, 1, 3]

    with torch.no_grad():
        for batch in loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)
            targets = batch["targets"].to(device)

            # Pass 1
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)

            # Pass 2
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

            # Extract scored columns and positions
            # Shape: (B, 68, 3)
            preds_scored = pred2[:, : Config.PRED_LEN, scored_cols_indices]
            targets_scored = targets[:, : Config.PRED_LEN, scored_cols_indices]

            # Squared Error
            se = (preds_scored - targets_scored) ** 2

            # Sum over batch and sequence length, keep column dimension
            sum_squared_errors += se.sum(dim=(0, 1))

            # Count total positions
            total_scored_positions += seq.size(0) * Config.PRED_LEN

    # Compute RMSE per column: sqrt(Sum_SE / N)
    rmse_per_col = torch.sqrt(sum_squared_errors / total_scored_positions)

    # MCRMSE is the mean of the column RMSEs
    global_mcrmse = torch.mean(rmse_per_col).item()

    return global_mcrmse


def train_and_predict():
    # Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Ensure directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)

    # 1. Data Loading
    print("Loading and processing data...")
    train_data = process_data("train", load_cached_data=True)
    val_data = process_data("val", load_cached_data=True)
    test_data = process_data("test", load_cached_data=True)

    train_dataset = RNADataset(train_data, "train")
    val_dataset = RNADataset(val_data, "val")
    test_dataset = RNADataset(test_data, "test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Initialization
    print("Initializing SDF-RN Model...")
    model = SDFRNModel().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.10f} | Val Loss: {val_loss:.10f}"
        )

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    print(f"Best Validation Loss: {best_val_loss:.10f}")

    # 4. Inference
    print("Generating predictions on test set...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    all_preds = []

    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            struct = batch["struct"].to(device)
            loop = batch["loop"].to(device)
            pid = batch["partner_id"].to(device)
            pidx = batch["partner_idx"].to(device)

            # Pass 1
            pred1 = model(seq, struct, loop, pid, pidx, prev_pred=None)
            # Pass 2
            pred2 = model(seq, struct, loop, pid, pidx, prev_pred=pred1)

            all_preds.append(pred2.cpu().numpy())

    # Concatenate predictions: (N_test, 107, 5)
    all_preds = np.concatenate(all_preds, axis=0)

    # 5. Submission Formatting
    print("Formatting submission...")
    test_ids = test_data["ids"]
    submission_rows = []

    # Config.ALL_TARGETS = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]
    # Model output corresponds to this order

    for i, sample_id in enumerate(test_ids):
        sample_pred = all_preds[i]  # (107, 5)
        for j in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{j}"
            vals = sample_pred[j]
            row_data = [row_id] + vals.tolist()
            submission_rows.append(row_data)

    columns = ["id_seqpos"] + Config.ALL_TARGETS
    sub_df = pd.DataFrame(submission_rows, columns=columns)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

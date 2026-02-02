import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from library.config import Config
from library.data import process_data, RNADataset
from library.model import StackingAwareHybridNet
from library.utils import mcrmse_loss


def set_seed(seed):
    """Sets the seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, device, mask):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for inputs, targets, _ in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        preds = model(inputs)

        # Calculate loss (Masked MCRMSE on scored columns)
        loss = mcrmse_loss(preds, targets, mask)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device, mask):
    """
    Validates the model and calculates Global MCRMSE.
    Accumulates SSE and counts across all batches before averaging.
    """
    model.eval()

    # Scored columns: reactivity (0), deg_Mg_pH10 (1), deg_Mg_50C (3)
    scored_indices = [0, 1, 3]

    # Accumulators for SSE per column
    total_sse = torch.zeros(len(scored_indices), device=device)
    total_count = 0

    with torch.no_grad():
        for inputs, targets, _ in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            preds = model(inputs)

            # Expand mask to match batch size: [Batch, Seq]
            batch_mask = mask.expand(inputs.size(0), -1)

            for i, col_idx in enumerate(scored_indices):
                p = preds[:, :, col_idx]
                t = targets[:, :, col_idx]

                # Squared Error masked
                se = (p - t) ** 2
                masked_se = se * batch_mask

                # Accumulate SSE for this column
                total_sse[i] += masked_se.sum()

            # Accumulate count of valid positions
            total_count += batch_mask.sum()

    # Calculate Global RMSE per column
    # RMSE = sqrt(Total SSE / Total Count)
    # Add epsilon to avoid division by zero
    rmse_per_col = torch.sqrt(total_sse / (total_count + 1e-8))

    # MCRMSE is the mean of the RMSEs of the scored columns
    global_mcrmse = rmse_per_col.mean().item()

    return global_mcrmse


def generate_submission(model_path, device):
    """
    Generates submission file using the best model.
    """
    print("Generating submission...")

    # Load test data
    test_inputs, _, test_ids = process_data(
        Config.TEST_CSV, "test", load_cached_data=True
    )

    # Dummy targets for dataset compatibility
    dummy_targets = np.zeros(
        (len(test_inputs), Config.SEQ_LEN, Config.OUTPUT_DIM), dtype=np.float32
    )

    test_dataset = RNADataset(test_inputs, dummy_targets, test_ids, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # Load Model
    model = StackingAwareHybridNet(Config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_list = []
    ids_list = []

    with torch.no_grad():
        for inputs, _, ids in test_loader:
            inputs = inputs.to(device)
            preds = model(inputs)  # [Batch, 107, 5]
            preds_list.append(preds.cpu().numpy())
            ids_list.extend(ids)

    all_preds = np.concatenate(preds_list, axis=0)

    # Format submission
    # id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
    submission_rows = []
    target_cols = ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        sample_preds = all_preds[i]  # [107, 5]
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            # Clip values to ensure no negative degradation if physically impossible,
            # though the metric allows negatives. Keeping raw predictions is usually safer.
            row_vals = sample_preds[seqpos].tolist()
            submission_rows.append([row_id] + row_vals)

    sub_df = pd.DataFrame(submission_rows, columns=["id_seqpos"] + target_cols)

    # Ensure directory exists
    os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run_training(
    epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, load_cached_data=True
):
    """
    Main training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # Ensure working directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # 1. Load Data
    train_inputs, train_targets, train_ids = process_data(
        Config.TRAIN_CSV, "train", load_cached_data=load_cached_data
    )
    val_inputs, val_targets, val_ids = process_data(
        Config.VAL_CSV, "val", load_cached_data=load_cached_data
    )

    train_dataset = RNADataset(train_inputs, train_targets, train_ids, mode="train")
    val_dataset = RNADataset(val_inputs, val_targets, val_ids, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Model Setup
    model = StackingAwareHybridNet(Config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    # 3. Mask Setup
    # Only the first 68 positions are scored
    mask = torch.zeros((1, Config.SEQ_LEN), device=device)
    mask[:, : Config.PRED_LEN] = 1.0

    # 4. Training Loop
    best_loss = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    patience = 7
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, mask)
        val_loss = validate(model, val_loader, device, mask)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_loss:.10f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val MCRMSE: {best_loss:.10f}")

    # 5. Generate Submission
    if os.path.exists(best_model_path):
        generate_submission(best_model_path, device)
    else:
        print("Error: Best model not found.")

    return best_model_path

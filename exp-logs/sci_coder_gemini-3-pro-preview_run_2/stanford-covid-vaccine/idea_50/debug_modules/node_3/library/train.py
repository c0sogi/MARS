import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from library.config import Config, seed_everything
from library.data import process_data, RNADataset
from library.model import REIDFN
from library.loss import MaskedMCRMSELoss


def train_one_epoch(model, loader, optimizer, criterion, device):
    """
    Executes one epoch of training using the REID-FN Two-Pass strategy.
    """
    model.train()
    total_loss = 0.0

    for x, pairs, y, _ in loader:
        x = x.to(device)
        pairs = pairs.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        # --- Pass 1: Zero Feedback ---
        # The model initializes feedback to zeros internally if prev_preds is None
        pred1 = model(x, pairs, prev_preds=None)

        # --- Pass 2: Feedback from Pass 1 ---
        # Detach pred1 to stop gradients flowing through the feedback generation of Pass 1
        # This stabilizes the recurrent loop training
        pred2 = model(x, pairs, prev_preds=pred1.detach())

        # --- Loss Calculation ---
        # MaskedMCRMSELoss handles slicing to Config.PRED_LEN and selecting scored columns
        loss1 = criterion(pred1, y)
        loss2 = criterion(pred2, y)

        # Weighted loss: prioritize the final output (Pass 2) but supervise the intermediate (Pass 1)
        loss = loss2 + 0.5 * loss1

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, device):
    """
    Computes the global MCRMSE on the validation set.
    """
    model.eval()

    # Scored columns indices in the 5-channel output: reactivity(0), deg_Mg_pH10(1), deg_Mg_50C(3)
    scored_indices = [0, 1, 3]

    total_se = 0.0
    total_count = 0

    with torch.no_grad():
        for x, pairs, y, _ in loader:
            x = x.to(device)
            pairs = pairs.to(device)
            y = y.to(device)

            # Two-pass inference
            pred1 = model(x, pairs, prev_preds=None)
            pred2 = model(x, pairs, prev_preds=pred1)

            # Slice to scored length and columns
            # pred2 shape: (B, 107, 5) -> slice to (B, 68, 5) -> select cols -> (B, 68, 3)
            pred_scored = pred2[:, : Config.PRED_LEN, scored_indices]
            target_scored = y[:, : Config.PRED_LEN, scored_indices]

            # Compute Squared Error
            se = (pred_scored - target_scored) ** 2
            total_se += se.sum().item()
            total_count += pred_scored.numel()  # Total number of elements (B * 68 * 3)

    # Global RMSE
    mse = total_se / (total_count + 1e-12)
    rmse = np.sqrt(mse)
    return rmse


def generate_submission(model, loader, device, output_path):
    """
    Generates predictions for the test set and saves to CSV.
    """
    model.eval()
    preds_list = []
    ids_list = []

    print("Generating predictions for test set...")
    with torch.no_grad():
        for x, pairs, ids in loader:
            x = x.to(device)
            pairs = pairs.to(device)

            # Two-pass inference
            pred1 = model(x, pairs, prev_preds=None)
            pred2 = model(x, pairs, prev_preds=pred1)

            preds_list.append(pred2.cpu().numpy())
            ids_list.extend(ids)

    # Concatenate all batches: (N_samples, 107, 5)
    preds_arr = np.concatenate(preds_list, axis=0)

    # Prepare submission rows
    submission_data = []

    # Columns in the output tensor correspond to Config.ALL_TARGET_COLS
    # ["reactivity", "deg_Mg_pH10", "deg_pH10", "deg_Mg_50C", "deg_50C"]

    for i, sample_id in enumerate(ids_list):
        # For each position in the sequence (length 107)
        for seqpos in range(Config.SEQ_LEN):
            row_id = f"{sample_id}_{seqpos}"
            row_values = preds_arr[i, seqpos].tolist()
            submission_data.append([row_id] + row_values)

    columns = ["id_seqpos"] + Config.ALL_TARGET_COLS
    submission_df = pd.DataFrame(submission_data, columns=columns)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def run_training(debug_size=None, epochs=Config.EPOCHS):
    """
    Main function to run the training pipeline.
    """
    seed_everything(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Data Loading
    train_dict, val_dict, test_dict = process_data(
        load_cached_data=True, debug_size=debug_size
    )

    train_ds = RNADataset(train_dict)
    val_ds = RNADataset(val_dict)
    test_ds = RNADataset(test_dict, is_test=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    # 2. Model Setup
    model = REIDFN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = MaskedMCRMSELoss().to(device)

    # 3. Training Loop
    best_score = float("inf")
    best_model_path = os.path.join(Config.CACHE_DIR, "best_model.pth")
    patience_counter = 0
    early_stopping_patience = 7

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_score = validate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        scheduler.step(val_score)

        if val_score < best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.10f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Validation Score: {best_score:.10f}")

    # 4. Inference
    if os.path.exists(best_model_path):
        print("Loading best model for inference...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found, using current model state.")

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

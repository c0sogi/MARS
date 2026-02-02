import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from library.config import (
    DEVICE,
    WORKING_DIR,
    SUBMISSION_PATH,
    ALL_TARGETS,
    SCORED_TARGETS,
    SEQ_LEN,
    PRED_LEN,
    BATCH_SIZE,
    LR,
    EPOCHS,
    PATIENCE,
    SEED,
)
from library.model import StagedInteractiveDenseNet
from library.data import get_loaders
from library.utils import GlobalMCRMSE, set_seed

# Constants for scoring
SEQ_SCORED = 68
SCORED_INDICES = [i for i, t in enumerate(ALL_TARGETS) if t in SCORED_TARGETS]


def criterion(preds, targets):
    """
    MCRMSE Loss computed only on the scored columns and scored sequence length.

    Args:
        preds: (Batch, SeqLen, NumTargets)
        targets: (Batch, SeqLen, NumTargets)
    """
    # Slice: First 68 positions, and only the specific columns (0, 1, 3)
    p = preds[:, :SEQ_SCORED, SCORED_INDICES]
    t = targets[:, :SEQ_SCORED, SCORED_INDICES]

    # MSE per element
    mse = (p - t) ** 2

    # Mean over batch and sequence length, then sqrt (RMSE)
    # We want mean columnwise RMSE.
    # MCRMSE = Mean(Sqrt(Mean(Error^2)))
    # For optimization, minimizing Mean(MSE) is generally sufficient and more stable,
    # but to align strictly with MCRMSE we can compute it explicitly.
    # Here we use standard MSE loss on the sliced data as a proxy for optimization
    # because optimizing MSE is equivalent to optimizing RMSE.
    loss = nn.functional.mse_loss(p, t)

    return loss


def train_fn(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_idx, (features, partners, targets) in enumerate(loader):
        features = features.to(device)
        partners = partners.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        outputs = model(features, partners)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def eval_fn(model, loader, device):
    model.eval()
    metric_calc = GlobalMCRMSE(device=device, seq_scored=SEQ_SCORED)

    with torch.no_grad():
        for features, partners, targets in loader:
            features = features.to(device)
            partners = partners.to(device)
            targets = targets.to(device)

            outputs = model(features, partners)

            # Update global metric accumulator
            metric_calc.update(outputs, targets)

    return metric_calc.compute()


def predict_fn(model, loader, device):
    model.eval()
    all_preds = []
    all_ids = []

    with torch.no_grad():
        for features, partners, ids in loader:
            features = features.to(device)
            partners = partners.to(device)

            outputs = model(features, partners)

            # Move to CPU and numpy
            preds_np = outputs.cpu().numpy()

            all_preds.append(preds_np)
            all_ids.extend(ids)

    # Concatenate all batches: (N_samples, SeqLen, NumTargets)
    all_preds = np.concatenate(all_preds, axis=0)
    return all_preds, all_ids


def run():
    # 1. Setup
    set_seed(SEED)
    os.makedirs(WORKING_DIR, exist_ok=True)
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Device: {DEVICE}")

    # 2. Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=True)

    # 3. Model & Optimizer
    model = StagedInteractiveDenseNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=False
    )

    # 4. Training Loop
    best_score = float("inf")
    early_stop_count = 0

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_fn(model, train_loader, optimizer, DEVICE)
        val_score = eval_fn(model, val_loader, DEVICE)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val MCRMSE: {val_score:.10f}"
        )

        # Save Best Model
        if val_score < best_score:
            best_score = val_score
            early_stop_count = 0
            torch.save(model.state_dict(), best_model_path)
            # print(f"  New best model saved! Score: {best_score:.6f}")
        else:
            early_stop_count += 1

        # Early Stopping
        if early_stop_count >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Best Validation Score: {best_score:.10f}")

    # 5. Inference
    print("Generating submission...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

    preds, ids = predict_fn(model, test_loader, DEVICE)

    # 6. Format Submission
    # preds shape: (N_samples, 107, 5)
    # We need to flatten this to (N_samples * 107, 6)
    # Columns: id_seqpos, reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C

    submission_data = []

    for i, sample_id in enumerate(ids):
        sample_preds = preds[i]  # (107, 5)

        for seqpos in range(PRED_LEN):
            # Row ID
            row_id = f"{sample_id}_{seqpos}"

            # Prediction values for this position
            # Order in ALL_TARGETS: reactivity, deg_Mg_pH10, deg_pH10, deg_Mg_50C, deg_50C
            vals = sample_preds[seqpos]

            row_data = {
                "id_seqpos": row_id,
                "reactivity": vals[0],
                "deg_Mg_pH10": vals[1],
                "deg_pH10": vals[2],
                "deg_Mg_50C": vals[3],
                "deg_50C": vals[4],
            }
            submission_data.append(row_data)

    # Create DataFrame
    submission_df = pd.DataFrame(submission_data)

    # Ensure column order matches sample submission
    cols = ["id_seqpos"] + ALL_TARGETS
    submission_df = submission_df[cols]

    # Save
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")


if __name__ == "__main__":
    run()

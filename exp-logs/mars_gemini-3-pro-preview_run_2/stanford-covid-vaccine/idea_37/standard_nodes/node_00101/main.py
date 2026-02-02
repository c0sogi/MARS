import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.stats as stats

# Import from provided library files
from library.config import (
    TRAIN_CSV,
    VAL_CSV,
    TEST_CSV,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    TARGET_COLS,
    SCORED_COLS,
    BATCH_SIZE,
    LEARNING_RATE,
    EPOCHS,
    WEIGHT_DECAY,
    GRAD_CLIP,
    PATIENCE,
    SEED,
    NUM_PASSES,
)
from library.utils import set_seed, mcrmse, format_submission
from library.data import get_train_loader, get_val_loader, get_test_loader
from library.model import IterativeRefinedDCN

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_scored_indices():
    """Returns indices of columns used for scoring."""
    return [i for i, col in enumerate(TARGET_COLS) if col in SCORED_COLS]


def criterion_mcrmse(pred, true):
    """
    Differentiable MCRMSE loss for training.
    Computes mean of column-wise RMSEs for scored columns.
    """
    scored_indices = get_scored_indices()

    # Select scored columns: (B, L, 5) -> (B, L, 3)
    pred_scored = pred[:, :, scored_indices]
    true_scored = true[:, :, scored_indices]

    # MSE per column (averaging over batch and sequence length)
    # Shape: (3,)
    mse_per_col = torch.mean((pred_scored - true_scored) ** 2, dim=(0, 1))

    # RMSE per column
    rmse_per_col = torch.sqrt(mse_per_col + 1e-8)  # Add epsilon for stability

    # Mean of RMSEs
    loss = torch.mean(rmse_per_col)
    return loss


def combined_loss(outputs, targets):
    """
    Computes combined loss: Loss(Pass2) + 0.5 * Loss(Pass1)
    """
    # outputs is a list [pred_pass1, pred_pass2]
    loss1 = criterion_mcrmse(outputs[0], targets)
    loss2 = criterion_mcrmse(outputs[1], targets)
    return loss2 + 0.5 * loss1


def train_one_epoch(model, loader, optimizer, clip_value):
    model.train()
    running_loss = 0.0

    for batch in loader:
        # Unpack batch
        # x: (B, L, FeatDim)
        # p_idx: (B, L)
        # p_mask: (B, L)
        # y: (B, L, 5)
        x, p_idx, p_mask, y = [t.to(device) for t in batch]

        optimizer.zero_grad()

        # Forward pass returns list of predictions from recycling steps
        outputs = model(x, p_idx, p_mask)

        loss = combined_loss(outputs, y)

        loss.backward()

        if clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in loader:
            x, p_idx, p_mask, y = [t.to(device) for t in batch]

            # Forward pass
            outputs = model(x, p_idx, p_mask)
            # Use final pass prediction for validation
            final_pred = outputs[-1]

            all_preds.append(final_pred.cpu())
            all_targets.append(y.cpu())

    # Concatenate
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)

    # Compute metric using utility function
    score = mcrmse(all_targets, all_preds)
    return score, all_preds, all_targets


def run_failure_analysis(val_preds, val_targets):
    """
    Correlates error magnitude with validation metadata features.
    """
    print("\n=== Failure Analysis ===")

    # 1. Calculate Error per Sample
    # val_preds, val_targets: (N, L, 5)
    scored_indices = get_scored_indices()

    vp = val_preds.numpy()[:, :, scored_indices]
    vt = val_targets.numpy()[:, :, scored_indices]

    # MSE per sample: (N,)
    # Average over Length and Scored Columns
    mse_per_sample = np.mean((vp - vt) ** 2, axis=(1, 2))
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 2. Load Metadata
    # We assume val_loader (shuffle=False) matches val.csv order
    if not os.path.exists(VAL_CSV):
        print("Validation CSV not found for analysis.")
        return

    val_df = pd.read_csv(VAL_CSV)

    if len(val_df) != len(rmse_per_sample):
        print(
            f"Warning: Mismatch in validation set size. DF: {len(val_df)}, Preds: {len(rmse_per_sample)}"
        )
        return

    # 3. Correlate
    val_df["model_error"] = rmse_per_sample

    # Features to check
    features = ["signal_to_noise", "seq_length", "mean_reactivity"]
    # Add derived features
    val_df["gc_content"] = val_df["sequence"].apply(
        lambda x: (x.count("G") + x.count("C")) / len(x)
    )
    val_df["paired_ratio"] = val_df["structure"].apply(
        lambda x: (x.count("(") + x.count(")")) / len(x)
    )

    features.extend(["gc_content", "paired_ratio"])

    print(f"{'Feature':<20} | {'Correlation with Error':<25}")
    print("-" * 50)

    for feat in features:
        if feat in val_df.columns:
            try:
                corr, _ = stats.pearsonr(val_df[feat], val_df["model_error"])
                print(f"{feat:<20} | {corr:.4f}")
            except Exception:
                pass
    print("-" * 50)


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Device: {device}")

    # 2. Data Loaders
    print("Loading data...")
    train_loader = get_train_loader(load_cached_data=True)
    val_loader = get_val_loader(load_cached_data=True)

    # 3. Model Initialization
    model = IterativeRefinedDCN().to(device)
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE)

    # 4. Training Loop
    best_val_score = float("inf")
    early_stop_counter = 0

    print(f"Starting training for {EPOCHS} epochs...")

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, GRAD_CLIP)
        val_score, _, _ = validate(model, val_loader)

        scheduler.step(val_score)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val MCRMSE: {val_score:.4f}"
        )

        if val_score < best_val_score:
            best_val_score = val_score
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if early_stop_counter >= PATIENCE + 2:  # Little extra buffer
            print("Early stopping triggered.")
            break

    # 5. Final Evaluation
    print("Loading best model for evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

    final_val_score, val_preds, val_targets = validate(model, val_loader)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_val_score}")

    # 6. Failure Analysis
    run_failure_analysis(val_preds, val_targets)

    # 7. Submission
    THRESHOLD = 0.47142532743789534

    if final_val_score < THRESHOLD:
        print("Validation score meets threshold. Generating submission...")

        test_loader, test_ids = get_test_loader(load_cached_data=True)
        model.eval()

        test_preds = []

        with torch.no_grad():
            for batch in test_loader:
                # Test loader yields (x, p_idx, p_mask, dummy_y)
                x, p_idx, p_mask, _ = [t.to(device) for t in batch]

                outputs = model(x, p_idx, p_mask)
                final_pred = outputs[-1]

                test_preds.append(final_pred.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        format_submission(test_ids, test_preds, SUBMISSION_PATH)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"Validation score {final_val_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

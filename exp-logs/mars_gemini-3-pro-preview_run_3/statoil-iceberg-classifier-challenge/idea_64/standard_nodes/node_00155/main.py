import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

from library.utils import (
    load_and_process_data,
    set_seed,
    logger,
    train_one_epoch,
    validate,
)
from library.data import get_fold_loaders, get_test_loader
from library.model import MSICNN


def run():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters for Fast Baseline
    EPOCHS = 50
    BATCH_SIZE = 32
    PATIENCE = 12

    # 2. Data Loading
    # Load cached data to save time
    data = load_and_process_data(load_cached_data=True)
    X = data["X_train"]
    y = data["y_train"]
    angles = data["angle_train"]

    # Containers for predictions
    oof_preds = np.zeros(len(X))
    test_preds_accum = np.zeros(len(data["X_test"]))

    # 3. Cross-Validation Loop
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        # Get loaders with leak-free angle imputation
        train_loader, val_loader, fold_median = get_fold_loaders(
            fold_index=fold,
            n_splits=5,
            batch_size=BATCH_SIZE,
            load_cached_data=True,
            seed=42,
        )

        # Initialize Model & Optimizer
        model = MSICNN().to(device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()

        best_loss = float("inf")
        best_state = None
        patience_counter = 0

        # Training
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_loss = validate(model, val_loader, criterion, device)

            # Simple logging
            # print(f"Fold {fold+1} Epoch {epoch+1} Train: {train_loss:.4f} Val: {val_loss:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        # Load best model for inference
        model.load_state_dict(best_state)
        model.eval()

        # Inference: Validation (OOF)
        val_preds_fold = []
        with torch.no_grad():
            for images, angs, _ in val_loader:
                images = images.to(device)
                angs = angs.to(device)
                # Model returns (B, 1) logits in eval mode
                logits = model(images, angs)
                probs = torch.sigmoid(logits)
                val_preds_fold.append(probs.cpu().numpy())

        oof_preds[val_idx] = np.concatenate(val_preds_fold).flatten()

        # Inference: Test
        # Use fold_median for test set angle imputation to match training distribution
        test_loader, _ = get_test_loader(
            batch_size=BATCH_SIZE, load_cached_data=True, angle_impute_val=fold_median
        )

        fold_test_preds = []
        with torch.no_grad():
            for images, angs in test_loader:
                images = images.to(device)
                angs = angs.to(device)
                logits = model(images, angs)
                probs = torch.sigmoid(logits)
                fold_test_preds.append(probs.cpu().numpy())

        test_preds_accum += np.concatenate(fold_test_preds).flatten() / 5.0

        # Cleanup
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Validation Assessment
    final_metric = log_loss(y, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(y - oof_preds)

    # Compute image statistics for correlation
    # X shape: (N, 3, 75, 75). Channel 0: Band 1, Channel 1: Band 2.
    b1_mean = np.mean(X[:, 0, :, :], axis=(1, 2))
    b1_std = np.std(X[:, 0, :, :], axis=(1, 2))
    b2_mean = np.mean(X[:, 1, :, :], axis=(1, 2))
    b2_std = np.std(X[:, 1, :, :], axis=(1, 2))

    # Handle NaNs in angles for correlation calculation
    valid_angles = angles[~np.isnan(angles)]
    global_median_angle = np.median(valid_angles) if len(valid_angles) > 0 else 0.0
    angles_filled = np.nan_to_num(angles, nan=global_median_angle)

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_filled,
            "b1_mean": b1_mean,
            "b1_std": b1_std,
            "b2_mean": b2_mean,
            "b2_std": b2_std,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Submission
    THRESHOLD = 0.17174082291273365
    if final_metric < THRESHOLD:
        _, test_ids = get_test_loader(batch_size=BATCH_SIZE, load_cached_data=True)
        submission = pd.DataFrame({"id": test_ids, "is_iceberg": test_preds_accum})

        os.makedirs("./submission", exist_ok=True)
        submission.to_csv("./submission/submission.csv", index=False)
        print(
            f"\nMetric {final_metric} < {THRESHOLD}. Submission saved to ./submission/submission.csv"
        )
    else:
        print(f"\nMetric {final_metric} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    run()

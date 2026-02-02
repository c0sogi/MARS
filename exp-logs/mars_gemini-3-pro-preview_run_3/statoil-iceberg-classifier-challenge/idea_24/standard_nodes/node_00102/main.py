import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from torchvision import transforms

# Import provided library functions
from library.utils import load_data, set_seed
from library.dataset import IcebergDataset
from library.model import CAMA_CNN, train_one_epoch, validate


def run():
    # --- Configuration ---
    SEED = 42
    N_FOLDS = 5
    EPOCHS = 20  # Fast baseline execution
    BATCH_SIZE = 32
    LR = 1e-3
    WEIGHT_DECAY = 1e-4
    PATIENCE = 5
    CACHE_DIR = "./working/idea_24"
    SUBMISSION_PATH = "./submission/submission.csv"
    THRESHOLD = 0.18120490171618245

    # --- Setup ---
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Data Loading ---
    # Load cached data using library function
    data = load_data(cache_dir=CACHE_DIR, load_cached_data=True)
    (
        X_train_raw,
        y_train_raw,
        angles_train_raw,
        X_val_raw,
        y_val_raw,
        angles_val_raw,
        X_test,
        ids_test,
        angles_test,
    ) = data

    # Combine Train and Val for Stratified K-Fold CV
    X_all = np.concatenate([X_train_raw, X_val_raw], axis=0)
    y_all = np.concatenate([y_train_raw, y_val_raw], axis=0)
    angles_all = np.concatenate([angles_train_raw, angles_val_raw], axis=0)

    # Placeholders for results
    oof_preds = np.zeros(len(y_all))
    test_preds_accum = np.zeros((len(X_test), N_FOLDS))

    # --- Cross-Validation Loop ---
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

    # Prepare Test Loader (reused across folds)
    test_dataset = IcebergDataset(X_test, angles_test, transform=None)
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    print(f"Starting {N_FOLDS}-Fold CV on {len(X_all)} samples...")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
        # Split Data
        X_train_fold, X_val_fold = X_all[train_idx], X_all[val_idx]
        y_train_fold, y_val_fold = y_all[train_idx], y_all[val_idx]
        angles_train_fold, angles_val_fold = angles_all[train_idx], angles_all[val_idx]

        # Augmentation for training
        train_transform = transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip()]
        )

        # Datasets & Loaders
        train_ds = IcebergDataset(
            X_train_fold, angles_train_fold, y_train_fold, transform=train_transform
        )
        val_ds = IcebergDataset(X_val_fold, angles_val_fold, y_val_fold, transform=None)

        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True
        )

        # Model, Optimizer, Criterion
        model = CAMA_CNN().to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion = nn.BCEWithLogitsLoss()

        # Training Loop
        best_val_loss = float("inf")
        patience_counter = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            _ = train_one_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds_fold, _ = validate(model, val_loader, criterion, device)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                break

        # Load best model for inference
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Generate OOF Predictions for this fold
        model.eval()
        # Re-run validation to get predictions from the best model state
        _, val_preds_fold, _ = validate(model, val_loader, criterion, device)
        oof_preds[val_idx] = val_preds_fold.flatten()

        # Generate Test Predictions
        fold_test_preds = []
        with torch.no_grad():
            for images, angles in test_loader:
                images = images.to(device)
                angles = angles.to(device)
                outputs = model(images, angles)
                probs = torch.sigmoid(outputs)
                fold_test_preds.extend(probs.cpu().numpy().flatten())

        test_preds_accum[:, fold] = fold_test_preds

    # --- Metric Calculation ---
    # Calculate Log Loss on the full Out-Of-Fold predictions
    final_metric = log_loss(y_all, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # --- Failure Analysis ---
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(oof_preds - y_all)

    # Calculate simple image statistics for correlation
    # X_all is (N, 3, 75, 75). Channel 0 is HH, Channel 1 is HV.
    b1_mean = X_all[:, 0, :, :].mean(axis=(1, 2))
    b2_mean = X_all[:, 1, :, :].mean(axis=(1, 2))

    analysis_df = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": angles_all,
            "band_1_mean": b1_mean,
            "band_2_mean": b2_mean,
        }
    )

    correlations = analysis_df.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # --- Submission ---
    if final_metric < THRESHOLD:
        print(
            f"\nMetric ({final_metric}) is better than threshold ({THRESHOLD}). Generating submission..."
        )

        # Average predictions across folds
        avg_test_preds = np.mean(test_preds_accum, axis=1)

        # Create submission DataFrame
        submission_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        submission_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nMetric ({final_metric}) did not meet threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    run()

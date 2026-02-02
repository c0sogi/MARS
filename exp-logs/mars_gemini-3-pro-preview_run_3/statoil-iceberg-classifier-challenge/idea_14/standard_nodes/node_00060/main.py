import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

# Import from library
from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    SEED,
    N_FOLDS,
    NUM_EPOCHS,
    PATIENCE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    DEVICE,
    NUM_WORKERS,
)
from library.utils import set_seed, save_checkpoint
from library.model import HybridWideSEResNet
from library.data_loader import get_dataloaders, IcebergDataset
from library.train_eval import get_transforms, train_one_epoch, evaluate, predict


def main():
    # 1. Setup
    set_seed(SEED)
    print(f"Running on device: {DEVICE}")

    # 2. Data Loading
    # Load cached data to speed up execution
    print("Loading data...")
    base_train_loader, base_val_loader, test_loader = get_dataloaders(
        load_cached_data=True, debug=False
    )

    # Extract arrays from the loaders' datasets to perform custom K-Fold splitting
    X_train = base_train_loader.dataset.X
    angles_train = base_train_loader.dataset.angles
    y_train = base_train_loader.dataset.y
    ids_train = base_train_loader.dataset.ids

    X_val = base_val_loader.dataset.X
    angles_val = base_val_loader.dataset.angles
    y_val = base_val_loader.dataset.y
    ids_val = base_val_loader.dataset.ids

    # Merge train and val sets for Stratified K-Fold
    X_full = np.concatenate([X_train, X_val], axis=0)
    angles_full = np.concatenate([angles_train, angles_val], axis=0)
    y_full = np.concatenate([y_train, y_val], axis=0)
    ids_full = np.concatenate([ids_train, ids_val], axis=0)

    print(f"Total training samples: {len(y_full)}")

    # 3. K-Fold Training
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    train_tf, eval_tf = get_transforms()

    fold_scores = []

    # Lists for failure analysis
    oof_preds = []
    oof_targets = []
    oof_angles = []
    oof_b1_means = []
    oof_b1_stds = []
    oof_b2_means = []
    oof_b2_stds = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_full, y_full)):
        print(f"\n--- Fold {fold + 1}/{N_FOLDS} ---")

        # Prepare Datasets for this fold
        train_ds = IcebergDataset(
            X_full[train_idx],
            angles_full[train_idx],
            y_full[train_idx],
            ids_full[train_idx],
            transform=train_tf,
        )
        val_ds = IcebergDataset(
            X_full[val_idx],
            angles_full[val_idx],
            y_full[val_idx],
            ids_full[val_idx],
            transform=eval_tf,
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Model Initialization
        model = HybridWideSEResNet().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Training Loop
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(NUM_EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss = evaluate(model, val_loader, criterion, DEVICE)

            # Save Checkpoint if best
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "state_dict": model.state_dict(),
                        "best_score": best_loss,
                    },
                    is_best=True,
                    checkpoint_dir=WORKING_DIR,
                    fold_idx=fold,
                )
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                # Early stopping
                break

        fold_scores.append(best_loss)
        print(f"Fold {fold+1} Best Log Loss: {best_loss}")

        # --- Failure Analysis Data Collection ---
        # Load best model for this fold
        best_ckpt = os.path.join(WORKING_DIR, f"model_best_fold_{fold}.pth")
        checkpoint = torch.load(best_ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        # Generate predictions on validation set
        fold_preds = []
        fold_targets = []

        with torch.no_grad():
            for batch in val_loader:
                imgs, angs, lbls, _ = batch
                imgs = imgs.to(DEVICE)
                angs = angs.to(DEVICE)

                out = model(imgs, angs)
                probs = torch.sigmoid(out).cpu().numpy().flatten()
                fold_preds.extend(probs)
                fold_targets.extend(lbls.numpy().flatten())

        oof_preds.extend(fold_preds)
        oof_targets.extend(fold_targets)

        # Collect metadata features from raw arrays
        val_X_raw = X_full[val_idx]  # (N, 75, 75, 3)
        val_angles_raw = angles_full[val_idx]

        oof_angles.extend(val_angles_raw)

        # Calculate image statistics (Band 1=Index 0, Band 2=Index 1)
        b1 = val_X_raw[:, :, :, 0]
        b2 = val_X_raw[:, :, :, 1]

        oof_b1_means.extend(np.mean(b1, axis=(1, 2)))
        oof_b1_stds.extend(np.std(b1, axis=(1, 2)))
        oof_b2_means.extend(np.mean(b2, axis=(1, 2)))
        oof_b2_stds.extend(np.std(b2, axis=(1, 2)))

    # 4. Final Metric
    avg_val_loss = np.mean(fold_scores)
    print(f"Final Validation Metric: {avg_val_loss}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)
    oof_angles = np.array(oof_angles)
    oof_b1_means = np.array(oof_b1_means)
    oof_b1_stds = np.array(oof_b1_stds)
    oof_b2_means = np.array(oof_b2_means)
    oof_b2_stds = np.array(oof_b2_stds)

    # Calculate error magnitude
    errors = np.abs(oof_targets - oof_preds)

    # Create DataFrame for correlation analysis
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": oof_angles,
            "b1_mean": oof_b1_means,
            "b1_std": oof_b1_stds,
            "b2_mean": oof_b2_means,
            "b2_std": oof_b2_stds,
        }
    )

    # Compute correlations
    correlations = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(correlations)

    # 6. Conditional Submission
    THRESHOLD = 0.18145903282502943
    if avg_val_loss < THRESHOLD:
        print(
            f"\nValidation metric {avg_val_loss} < {THRESHOLD}. Generating submission..."
        )

        test_preds_sum = np.zeros(len(test_loader.dataset))
        test_ids = None

        # Ensemble predictions from all folds
        for fold in range(N_FOLDS):
            model = HybridWideSEResNet().to(DEVICE)
            ckpt_path = os.path.join(WORKING_DIR, f"model_best_fold_{fold}.pth")
            checkpoint = torch.load(ckpt_path, map_location=DEVICE)
            model.load_state_dict(checkpoint["state_dict"])

            # Use library predict function
            ids, preds = predict(model, test_loader, DEVICE)
            test_preds_sum += preds
            if test_ids is None:
                test_ids = ids

        # Average predictions
        avg_preds = test_preds_sum / N_FOLDS

        # Save submission
        df_sub = pd.DataFrame({"id": test_ids, "is_iceberg": avg_preds})
        df_sub.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(
            f"\nValidation metric {avg_val_loss} >= {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

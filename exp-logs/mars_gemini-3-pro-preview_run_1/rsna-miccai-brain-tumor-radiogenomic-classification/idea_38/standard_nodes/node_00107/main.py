import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

# Import from provided library files
from library.config import (
    SEED,
    seed_everything,
    DEVICE,
    NUM_WORKERS,
    BATCH_SIZE,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
)
from library.utils import load_checkpoint
from library.dataset import get_datasets, VAADataset, get_transforms
from library.model import build_model
from library.engine import train_model


def main():
    # 1. Initialization
    # Set seeds for reproducibility
    seed_everything(SEED)

    # Configuration for Fast Baseline
    # We use 15 epochs which is sufficient for the small dataset size (~500 samples)
    # to converge or trigger early stopping.
    EPOCHS = 15
    FOLDS = 5
    PATIENCE = 5

    print("Initializing Verified Anatomically-Anchored (VAA) Pipeline...")

    # 2. Data Loading
    # Load datasets using the library function.
    # load_cached_data=True ensures we use the pre-processed numpy arrays if available.
    train_ds_raw, val_ds_raw, test_ds = get_datasets(load_cached_data=True)

    # Merge provided train and val splits to perform our own 5-Fold CV
    # This maximizes data usage and provides a more robust validation metric.
    all_images = np.concatenate([train_ds_raw.images, val_ds_raw.images], axis=0)
    all_labels = np.concatenate([train_ds_raw.labels, val_ds_raw.labels], axis=0)
    all_ids = np.concatenate([train_ds_raw.ids, val_ds_raw.ids], axis=0)

    print(f"Total Combined Training Samples: {len(all_labels)}")
    print(f"Test Samples: {len(test_ds)}")

    # 3. Cross-Validation
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    # Arrays to store Out-Of-Fold predictions and targets
    oof_preds = np.zeros(len(all_labels))
    oof_targets = np.zeros(len(all_labels))

    # Accumulator for Test set predictions (Ensemble averaging)
    test_preds_accum = np.zeros(len(test_ds))

    # Create Test Loader once (used in every fold)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Iterate through Folds
    for fold, (train_idx, val_idx) in enumerate(skf.split(all_images, all_labels)):
        print(f"\n=== Fold {fold+1}/{FOLDS} ===")

        # Split Data
        X_train, X_val = all_images[train_idx], all_images[val_idx]
        y_train, y_val = all_labels[train_idx], all_labels[val_idx]
        ids_train, ids_val = all_ids[train_idx], all_ids[val_idx]

        # Create Dataset Objects
        train_dataset = VAADataset(
            X_train, y_train, ids_train, transform=get_transforms("train")
        )
        val_dataset = VAADataset(
            X_val, y_val, ids_val, transform=get_transforms("valid")
        )

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Build Model
        model = build_model()
        model = model.to(DEVICE)

        # Optimizer & Loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
        criterion = nn.BCEWithLogitsLoss()

        # Train Loop
        # train_model handles training, validation, early stopping, and saving the best checkpoint
        best_fold_score = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            DEVICE,
            num_epochs=EPOCHS,
            patience=PATIENCE,
        )

        # Reload Best Model for Inference
        # train_model saves to MODEL_SAVE_PATH. We must reload it to ensure we use the best weights.
        model, _ = load_checkpoint(model, MODEL_SAVE_PATH, device=DEVICE)
        model.eval()

        # Validation Inference (OOF)
        val_probs = []
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch["image"].to(DEVICE)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                val_probs.extend(probs)

        # Store OOF predictions
        oof_preds[val_idx] = val_probs
        oof_targets[val_idx] = y_val

        # Test Inference (Accumulate for Ensemble)
        fold_test_probs = []
        with torch.no_grad():
            for batch in test_loader:
                imgs = batch["image"].to(DEVICE)
                logits = model(imgs)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                fold_test_probs.extend(probs)

        test_preds_accum += np.array(fold_test_probs)

        # Cleanup to free GPU memory
        del model, optimizer, criterion, train_loader, val_loader
        torch.cuda.empty_cache()

    # 4. Evaluation
    # Calculate ROC AUC on the full OOF set
    final_auc = roc_auc_score(all_labels, oof_preds)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_auc}")

    # 5. Failure Analysis
    print("\n=== Failure Analysis ===")
    # Calculate absolute error magnitude
    errors = np.abs(all_labels - oof_preds)

    # Calculate simple image statistics to correlate with error
    # all_images shape is (N, H, W, 3). Channels: FLAIR, T1wCE, T2w
    print("Computing image statistics for analysis...")

    # Reshape to (N, Pixels, Channels) to compute stats per image per channel
    flat_images = all_images.reshape(all_images.shape[0], -1, 3)

    # Compute Mean and Std Dev for each channel
    mean_intensity = flat_images.mean(axis=1)  # (N, 3)
    std_intensity = flat_images.std(axis=1)  # (N, 3)

    # Create Analysis DataFrame
    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "mean_flair": mean_intensity[:, 0],
            "mean_t1wce": mean_intensity[:, 1],
            "mean_t2w": mean_intensity[:, 2],
            "std_flair": std_intensity[:, 0],
            "std_t1wce": std_intensity[:, 1],
            "std_t2w": std_intensity[:, 2],
        }
    )

    # Compute Correlation
    corr = df_analysis.corr()["error"].sort_values(ascending=False)
    print("Correlation between Error Magnitude and Input Features:")
    print(corr)

    # 6. Submission Logic
    THRESHOLD = 0.6705454545454544

    if final_auc > THRESHOLD:
        print(
            f"\nValidation metric {final_auc} > {THRESHOLD}. Generating submission..."
        )

        # Average predictions across folds
        avg_test_preds = test_preds_accum / FOLDS

        # Create submission DataFrame
        # test_ds.ids contains the BraTS21IDs corresponding to the predictions
        sub_df = pd.DataFrame({"BraTS21ID": test_ds.ids, "MGMT_value": avg_test_preds})

        # Ensure directory exists
        os.makedirs(os.path.dirname(SUBMISSION_PATH), exist_ok=True)

        # Save
        sub_df.to_csv(SUBMISSION_PATH, index=False)
        print(f"Submission saved to {SUBMISSION_PATH}")
    else:
        print(f"\nValidation metric {final_auc} <= {THRESHOLD}. Skipping submission.")


if __name__ == "__main__":
    main()

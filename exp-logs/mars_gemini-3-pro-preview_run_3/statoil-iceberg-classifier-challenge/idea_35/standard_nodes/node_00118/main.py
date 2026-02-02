import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss

from library.utils import set_seed
from library.dataset import process_split, IcebergDataset, get_transforms
from library.model import DPACNN
from library.trainer import run_fold


def predict(model, loader, device):
    """
    Generates predictions using a trained model.
    Returns a numpy array of probabilities.
    """
    model.eval()
    all_preds = []
    with torch.no_grad():
        # Loader returns (images, angles, labels/ids)
        # We only need images and angles for inference
        for images, angles, _ in loader:
            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def main():
    # 1. Setup
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    META_DIR = "./metadata"
    INPUT_DIR = "./input"
    SUBMISSION_DIR = "./submission"
    CHECKPOINT_DIR = "./working/checkpoints"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # 2. Load Data
    # Load Train (for Cross-Validation)
    print("Loading training data...")
    X_train_full, ang_train_full, y_train_full, ids_train_full = process_split(
        os.path.join(META_DIR, "train.csv"),
        os.path.join(INPUT_DIR, "train.json"),
        "train",
        load_cached_data=True,
    )

    # Load Holdout Validation (for Final Metric)
    print("Loading holdout validation data...")
    X_holdout, ang_holdout, y_holdout, ids_holdout = process_split(
        os.path.join(META_DIR, "val.csv"),
        os.path.join(INPUT_DIR, "train.json"),
        "val",
        load_cached_data=True,
    )

    # Load Test (for Submission)
    print("Loading test data...")
    X_test, ang_test, y_test, ids_test = process_split(
        os.path.join(META_DIR, "test.csv"),
        os.path.join(INPUT_DIR, "test.json"),
        "test",
        load_cached_data=True,
    )

    # 3. Imputation
    # Calculate median angle from the training set
    median_angle = np.nanmedian(ang_train_full)

    # Fill NaNs in all datasets
    ang_train_full = np.where(np.isnan(ang_train_full), median_angle, ang_train_full)
    ang_holdout = np.where(np.isnan(ang_holdout), median_angle, ang_holdout)
    ang_test = np.where(np.isnan(ang_test), median_angle, ang_test)

    # 4. 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    models = []

    # Prepare Holdout Loader (for evaluation)
    holdout_ds = IcebergDataset(
        X_holdout, ang_holdout, ids_holdout, y_holdout, transform=get_transforms("val")
    )
    holdout_loader = DataLoader(holdout_ds, batch_size=32, shuffle=False, num_workers=2)

    # Prepare Test Loader (for submission)
    test_ds = IcebergDataset(
        X_test, ang_test, ids_test, y=None, transform=get_transforms("test")
    )
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2)

    print(f"\nStarting 5-Fold CV on {len(X_train_full)} samples...")

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_full)):
        print(f"\n--- Fold {fold} ---")

        # Split data for this fold
        X_tr = X_train_full[train_idx]
        ang_tr = ang_train_full[train_idx]
        y_tr = y_train_full[train_idx]
        ids_tr = ids_train_full[train_idx]

        X_val = X_train_full[val_idx]
        ang_val = ang_train_full[val_idx]
        y_val = y_train_full[val_idx]
        ids_val = ids_train_full[val_idx]

        # Create Datasets
        train_ds = IcebergDataset(
            X_tr, ang_tr, ids_tr, y_tr, transform=get_transforms("train")
        )
        val_ds = IcebergDataset(
            X_val, ang_val, ids_val, y_val, transform=get_transforms("val")
        )

        # Create Loaders
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True
        )

        # Train
        model, best_score = run_fold(
            train_loader,
            val_loader,
            fold_idx=fold,
            epochs=75,
            patience=12,
            learning_rate=1e-3,
            weight_decay=1e-2,
            device=device,
            checkpoint_dir=CHECKPOINT_DIR,
        )
        models.append(model)

    # 5. Ensemble Evaluation on Holdout
    print("\nEvaluating Ensemble on Holdout Set...")
    holdout_preds = np.zeros((len(X_holdout), 1))

    for model in models:
        preds = predict(model, holdout_loader, device)
        holdout_preds += preds

    # Average predictions
    holdout_preds /= len(models)

    # Compute Log Loss (clipping to avoid undefined log(0))
    holdout_preds_clipped = np.clip(holdout_preds, 1e-15, 1 - 1e-15)
    final_log_loss = log_loss(y_holdout, holdout_preds_clipped)

    print(f"Final Validation Metric: {final_log_loss:.10f}")

    # 6. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate absolute error
    errors = np.abs(y_holdout.flatten() - holdout_preds.flatten())

    # Extract features for correlation analysis
    # Feature 1: Incidence Angle
    feat_angle = ang_holdout
    # Feature 2: Band 1 Mean (HH)
    feat_b1_mean = np.mean(X_holdout[:, 0, :, :], axis=(1, 2))
    # Feature 3: Band 2 Mean (HV)
    feat_b2_mean = np.mean(X_holdout[:, 1, :, :], axis=(1, 2))

    df_analysis = pd.DataFrame(
        {
            "error": errors,
            "inc_angle": feat_angle,
            "b1_mean": feat_b1_mean,
            "b2_mean": feat_b2_mean,
        }
    )

    # Calculate correlation
    corrs = df_analysis.corr()["error"].drop("error")
    print("Correlation between Error Magnitude and Features:")
    print(corrs)

    # 7. Submission
    THRESHOLD = 0.1806015565870406

    if final_log_loss < THRESHOLD:
        print(
            f"\nMetric ({final_log_loss:.6f}) is better than threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        test_preds = np.zeros((len(X_test), 1))
        for model in models:
            preds = predict(model, test_loader, device)
            test_preds += preds

        # Average predictions
        test_preds /= len(models)
        test_preds = test_preds.flatten()

        # Create submission DataFrame
        df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": test_preds})

        sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        df_sub.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")
    else:
        print(
            f"\nMetric ({final_log_loss:.6f}) did not meet threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()

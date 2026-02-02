import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss

# Import provided library functions
from library.utils import seed_everything, get_device
from library.dataset import load_data, IcebergDataset, get_transforms
from library.engine import fit_fold
from library.model import SimpleCNN


def predict_dataset(model, X, angles, ids_or_y, device, batch_size=32):
    """
    Helper function to generate predictions for a dataset.
    """
    dataset = IcebergDataset(
        X, angles, ids_or_y, transform=get_transforms("test"), mode="test"
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    model.eval()
    preds = []

    with torch.no_grad():
        for images, angs, _ in loader:
            images = images.to(device)
            angs = angs.to(device)

            outputs = model(images, angs)
            probs = torch.sigmoid(outputs)
            preds.append(probs.cpu().numpy())

    return np.vstack(preds)


def run_pipeline():
    # 1. Setup
    seed_everything(42)
    device = get_device()

    working_dir = "./working/optimized"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print("Loading Data...")
    # Load separate train and val sets (Cite Lesson 00044: Use fixed hold-out set)
    X_train, angles_train, y_train = load_data("train", load_cached_data=True)
    X_holdout, angles_holdout, y_holdout = load_data("val", load_cached_data=True)

    # Load Test Data
    X_test, angles_test, ids_test = load_data("test", load_cached_data=True)

    print(f"Training Set: {len(y_train)} samples")
    print(f"Hold-Out Validation Set: {len(y_holdout)} samples")
    print(f"Test Set: {len(ids_test)} samples")

    # 2. Cross-Validation Training on Train Set
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Accumulators for Hold-Out and Test predictions
    holdout_preds_accum = np.zeros((len(y_holdout), 1))
    test_preds_accum = np.zeros((len(ids_test), 1))

    # Training parameters
    epochs = 30
    batch_size = 32
    patience = 8

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        print(f"\n--- Fold {fold} ---")

        # Split Data (Internal CV for early stopping)
        X_tr, X_va = X_train[train_idx], X_train[val_idx]
        a_tr, a_va = angles_train[train_idx], angles_train[val_idx]
        y_tr, y_va = y_train[train_idx], y_train[val_idx]

        # Train
        fit_fold(
            fold=fold,
            X_train=X_tr,
            angles_train=a_tr,
            y_train=y_tr,
            X_val=X_va,
            angles_val=a_va,
            y_val=y_va,
            epochs=epochs,
            batch_size=batch_size,
            patience=patience,
            save_dir=working_dir,
        )

        # Load Best Model
        model_path = os.path.join(working_dir, f"model_fold_{fold}.pth")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Predict on Hold-Out Set (Ensemble component)
        fold_holdout_probs = predict_dataset(
            model, X_holdout, angles_holdout, y_holdout, device
        )
        holdout_preds_accum += fold_holdout_probs

        # Predict on Test Set (Ensemble component)
        fold_test_probs = predict_dataset(model, X_test, angles_test, ids_test, device)
        test_preds_accum += fold_test_probs

    # Average predictions (Ensembling)
    avg_holdout_preds = holdout_preds_accum / n_folds
    avg_test_preds = test_preds_accum / n_folds

    # 3. Validation Metric (Cite Lesson 00044: Compare Hold-Out Ensemble metric)
    # Clip predictions to avoid log(0) errors
    avg_holdout_preds_clipped = np.clip(avg_holdout_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_holdout, avg_holdout_preds_clipped)

    print(f"\nFinal Validation Metric (Hold-Out Ensemble): {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_holdout.reshape(-1, 1) - avg_holdout_preds)

    # Extract features for correlation
    # 1. Incidence Angle
    feat_angle = angles_holdout.reshape(-1, 1)

    # 2. Band Mean Intensities
    feat_b1_mean = np.mean(X_holdout[:, :, :, 0], axis=(1, 2)).reshape(-1, 1)
    feat_b2_mean = np.mean(X_holdout[:, :, :, 1], axis=(1, 2)).reshape(-1, 1)

    # Calculate correlations
    corr_angle = np.corrcoef(errors.flatten(), feat_angle.flatten())[0, 1]
    corr_b1 = np.corrcoef(errors.flatten(), feat_b1_mean.flatten())[0, 1]
    corr_b2 = np.corrcoef(errors.flatten(), feat_b2_mean.flatten())[0, 1]

    print("Correlation between Error Magnitude and Input Features:")
    print(f"  Incidence Angle: {corr_angle:.4f}")
    print(f"  Band 1 Mean (HH): {corr_b1:.4f}")
    print(f"  Band 2 Mean (HV): {corr_b2:.4f}")

    # 5. Submission
    threshold = 0.18145903282502943
    if final_metric < threshold:
        print(f"\nMetric meets threshold ({threshold}). Generating submission...")

        submission_df = pd.DataFrame(
            {"id": ids_test, "is_iceberg": avg_test_preds.flatten()}
        )

        save_path = os.path.join(submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    run_pipeline()

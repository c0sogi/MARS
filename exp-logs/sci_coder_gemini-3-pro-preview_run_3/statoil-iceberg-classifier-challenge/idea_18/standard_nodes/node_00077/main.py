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
from library.model import APCNN


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

    working_dir = "./working/idea_18"
    submission_dir = "./submission"
    os.makedirs(working_dir, exist_ok=True)
    os.makedirs(submission_dir, exist_ok=True)

    print("Loading Data...")
    # Load separate train and val sets from metadata and combine for CV
    X_train_part, angles_train_part, y_train_part = load_data(
        "train", load_cached_data=True
    )
    X_val_part, angles_val_part, y_val_part = load_data("val", load_cached_data=True)

    # Concatenate to form full dev set
    X_dev = np.concatenate([X_train_part, X_val_part], axis=0)
    angles_dev = np.concatenate([angles_train_part, angles_val_part], axis=0)
    y_dev = np.concatenate([y_train_part, y_val_part], axis=0)

    # Load Test Data
    X_test, angles_test, ids_test = load_data("test", load_cached_data=True)

    print(f"Development Set: {len(y_dev)} samples")
    print(f"Test Set: {len(ids_test)} samples")

    # 2. Cross-Validation
    n_folds = 5
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    oof_preds = np.zeros((len(y_dev), 1))
    test_preds_accum = np.zeros((len(ids_test), 1))

    # Training parameters
    epochs = 30
    batch_size = 32
    patience = 8

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev, y_dev)):
        print(f"\n--- Fold {fold} ---")

        # Split Data
        X_tr, X_va = X_dev[train_idx], X_dev[val_idx]
        a_tr, a_va = angles_dev[train_idx], angles_dev[val_idx]
        y_tr, y_va = y_dev[train_idx], y_dev[val_idx]

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
        model = APCNN().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # Predict OOF
        val_probs = predict_dataset(model, X_va, a_va, y_va, device)
        oof_preds[val_idx] = val_probs

        # Predict Test
        fold_test_probs = predict_dataset(model, X_test, angles_test, ids_test, device)
        test_preds_accum += fold_test_probs

    # Average test predictions
    avg_test_preds = test_preds_accum / n_folds

    # 3. Validation Metric
    # Clip predictions to avoid log(0) errors, though sklearn handles it usually
    oof_preds_clipped = np.clip(oof_preds, 1e-15, 1 - 1e-15)
    final_metric = log_loss(y_dev, oof_preds_clipped)

    print(f"\nFinal Validation Metric: {final_metric}")

    # 4. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error magnitude
    errors = np.abs(y_dev.reshape(-1, 1) - oof_preds)

    # Extract features for correlation
    # 1. Incidence Angle
    feat_angle = angles_dev.reshape(-1, 1)

    # 2. Band Mean Intensities (Band 1 is index 0, Band 2 is index 1)
    # X_dev shape is (N, 75, 75, 3)
    feat_b1_mean = np.mean(X_dev[:, :, :, 0], axis=(1, 2)).reshape(-1, 1)
    feat_b2_mean = np.mean(X_dev[:, :, :, 1], axis=(1, 2)).reshape(-1, 1)

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

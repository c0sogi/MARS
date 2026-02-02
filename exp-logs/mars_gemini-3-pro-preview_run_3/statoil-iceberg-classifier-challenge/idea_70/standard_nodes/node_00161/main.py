import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

from library.config import Config
from library.utils import set_seed, load_checkpoint
from library.dataset import preprocess_data, get_fold_datasets, get_test_dataset
from library.model import AGICNN
from library.train import run_fold


def predict_loader(model, loader, device, has_labels=True):
    """
    Runs inference on a DataLoader.
    """
    model.eval()
    all_preds = []

    with torch.no_grad():
        for batch in loader:
            if has_labels:
                images, angles, _ = batch
            else:
                images, angles, _ = batch

            images = images.to(device)
            angles = angles.to(device)

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())

    return np.concatenate(all_preds)


def analyze_failures(y_true, y_pred, X, angles):
    """
    Performs failure analysis by correlating error magnitude with input features.
    """
    print("\n=== Failure Analysis ===")

    # Calculate error magnitude
    errors = np.abs(y_true - y_pred)

    # Feature 1: Incidence Angle
    # Handle NaNs in angles if any remain (though they should be imputed by now in the loader,
    # here we are using the raw array which might have NaNs, so we mask them)
    valid_angle_mask = ~np.isnan(angles)
    if np.sum(valid_angle_mask) > 0:
        corr_angle, _ = pearsonr(errors[valid_angle_mask], angles[valid_angle_mask])
        print(f"Correlation (Error vs Inc Angle): {corr_angle:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (All angles NaN)")

    # Feature 2: Image Brightness (Mean of Band 1 & 2)
    # X shape: (N, 3, 75, 75). Channels: HH, HV, Avg.
    # We'll use channel 0 (HH) and 1 (HV) means.
    img_means = np.mean(X, axis=(1, 2, 3))
    corr_mean, _ = pearsonr(errors, img_means)
    print(f"Correlation (Error vs Image Mean Intensity): {corr_mean:.4f}")

    # Feature 3: Image Contrast (Std of Band 1 & 2)
    img_stds = np.std(X, axis=(1, 2, 3))
    corr_std, _ = pearsonr(errors, img_stds)
    print(f"Correlation (Error vs Image Contrast/Std): {corr_std:.4f}")

    # Top failures
    worst_indices = np.argsort(errors)[-5:][::-1]
    print("Top 5 Worst Predictions (Index | True | Pred | Error):")
    for idx in worst_indices:
        print(f"  {idx:4d} | {y_true[idx]:.0f} | {y_pred[idx]:.4f} | {errors[idx]:.4f}")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading data...")
    X_train, y_train, angles_train, ids_train, X_test, angles_test, ids_test = (
        preprocess_data(load_cached_data=True)
    )

    # Placeholder for OOF predictions
    # We need to map predictions back to their original indices.
    # Since StratifiedKFold shuffles, we'll fill this array using the validation indices.
    oof_preds = np.zeros(len(y_train))

    # 3. Training Loop
    print(f"Starting {Config.NUM_FOLDS}-Fold Cross-Validation...")

    for fold in range(Config.NUM_FOLDS):
        print(f"\n--- Fold {fold} ---")

        # Train the model for this fold
        # run_fold saves the best checkpoint to disk
        run_fold(fold, X_train, y_train, angles_train, ids_train)

        # Reload best model for inference
        model = AGICNN()
        model.to(device)
        load_checkpoint(model, fold, load_best=True)

        # Get validation data for this fold to generate OOF preds
        _, val_ds = get_fold_datasets(
            X_train,
            y_train,
            angles_train,
            ids_train,
            fold=fold,
            num_folds=Config.NUM_FOLDS,
            seed=Config.SEED,
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        # Predict
        fold_preds = predict_loader(model, val_loader, device, has_labels=True)

        # Map back to original indices
        # We need to know which indices belonged to this validation fold.
        # We can reconstruct the split using the same seed.
        from sklearn.model_selection import StratifiedKFold

        skf = StratifiedKFold(
            n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED
        )
        splits = list(skf.split(X_train, y_train.astype(int)))
        _, val_idx = splits[fold]

        oof_preds[val_idx] = fold_preds

    # 4. Global Validation Metric
    final_metric = log_loss(y_train, oof_preds)
    print(f"\nFinal Validation Metric: {final_metric}")

    # 5. Failure Analysis
    analyze_failures(y_train, oof_preds, X_train, angles_train)

    # 6. Submission
    THRESHOLD = 0.17174082291273365

    if final_metric < THRESHOLD:
        print("\nMetric check passed. Generating submission...")

        # Prepare Test Loader
        # We pass full training angles to calculate global median for imputation
        test_ds = get_test_dataset(X_test, angles_test, ids_test, angles_train)
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        test_preds_sum = np.zeros(len(X_test))

        # Ensemble Inference
        for fold in range(Config.NUM_FOLDS):
            print(f"Inference with model fold {fold}...")
            model = AGICNN()
            model.to(device)
            load_checkpoint(model, fold, load_best=True)

            fold_test_preds = predict_loader(
                model, test_loader, device, has_labels=False
            )
            test_preds_sum += fold_test_preds

        avg_test_preds = test_preds_sum / Config.NUM_FOLDS

        # Create Submission DataFrame
        sub_df = pd.DataFrame({"id": ids_test, "is_iceberg": avg_test_preds})

        save_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")

    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()

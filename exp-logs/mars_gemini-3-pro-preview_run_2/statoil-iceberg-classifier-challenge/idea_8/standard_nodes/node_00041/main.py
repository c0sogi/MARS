import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything
from library.data_loader import get_loader
from library.model import MSAHN
from library.train_eval import run_fold


def predict_ensemble(loader, models, device):
    """
    Performs inference using an ensemble of models.
    Returns averaged probabilities and, if available, targets and metadata stats.
    """
    avg_preds = None
    targets = []
    inc_angles_list = []
    img_means_list = []

    # We iterate through the loader once to get data/metadata
    # Then pass through all models.
    # To save memory/time, we can iterate loader inside, but for ensemble
    # it is often easier to predict fold-by-fold and sum.

    # However, to avoid storing all images in memory, we process batch by batch
    # and run all models on that batch.

    for model in models:
        model.eval()

    batch_preds = []

    with torch.no_grad():
        for batch_data in loader:
            # Unpack data based on whether labels exist
            if len(batch_data) == 3:
                images, inc_angles, labels = batch_data
                targets.append(labels.cpu().numpy())
            else:
                images, inc_angles = batch_data

            images = images.to(device)
            inc_angles = inc_angles.to(device)

            # Store metadata for failure analysis (only needed once)
            if avg_preds is None:
                inc_angles_list.append(inc_angles.cpu().numpy())
                # Calculate mean intensity per image (across channels/pixels) for analysis
                # images shape: (B, 3, 75, 75)
                img_means_list.append(images.mean(dim=(1, 2, 3)).cpu().numpy())

            # Ensemble prediction for this batch
            batch_fold_preds = []
            for model in models:
                outputs = model(images, inc_angles)
                probs = torch.sigmoid(outputs)
                batch_fold_preds.append(probs.cpu().numpy())

            # Average across models for this batch
            # shape: (n_models, batch_size, 1) -> mean -> (batch_size, 1)
            mean_prob = np.mean(np.stack(batch_fold_preds), axis=0)
            batch_preds.append(mean_prob)

    # Concatenate all batches
    final_preds = np.concatenate(batch_preds, axis=0).flatten()

    final_targets = None
    if targets:
        final_targets = np.concatenate(targets, axis=0).flatten()

    final_inc = np.concatenate(inc_angles_list, axis=0).flatten()
    final_img_means = np.concatenate(img_means_list, axis=0).flatten()

    return final_preds, final_targets, final_inc, final_img_means


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Ensure directories exist
    Config.setup()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # 2. Load Metadata
    print("Loading metadata...")
    df_train_full = pd.read_csv(Config.TRAIN_META_FILE)
    df_holdout_val = pd.read_csv(Config.VAL_META_FILE)
    df_test = pd.read_csv(Config.TEST_META_FILE)

    # 3. Stratified 5-Fold Cross-Validation Training
    print("Starting 5-Fold Cross-Validation Training...")

    skf = StratifiedKFold(
        n_splits=Config.N_FOLDS, shuffle=True, random_state=Config.SEED
    )

    # We split the 'train.csv' data.
    # Note: df_train_full contains the indices relative to itself.
    X = df_train_full["id"].values
    y = df_train_full["is_iceberg"].values

    trained_model_paths = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        print(f"\n--- Fold {fold_idx} ---")

        # Create fold-specific DataFrames
        fold_train_df = df_train_full.iloc[train_idx].copy()
        fold_val_df = df_train_full.iloc[val_idx].copy()

        # Run training for this fold
        # run_fold saves the model and returns best metrics
        run_fold(fold_idx, fold_train_df, fold_val_df)

        # Construct path to the saved best model
        # run_fold saves to: Config.MODEL_CHECKPOINT_PREFIX + f"_{fold_idx}.pth"
        # and copies best to: ..._best.pth
        ckpt_path = f"{Config.MODEL_CHECKPOINT_PREFIX}_{fold_idx}_best.pth"
        trained_model_paths.append(ckpt_path)

    # 4. Evaluation on Hold-out Validation Set
    print("\nEvaluating Ensemble on Hold-out Validation Set...")

    # Load all models
    models = []
    for path in trained_model_paths:
        model = MSAHN().to(device)
        checkpoint = torch.load(path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        models.append(model)

    # Create DataLoader for Hold-out Validation
    val_loader = get_loader(
        df_holdout_val,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        augment=False,
        load_cached_data=True,
    )

    # Predict
    val_preds, val_targets, val_inc, val_img_means = predict_ensemble(
        val_loader, models, device
    )

    # Calculate Metric
    final_log_loss = log_loss(val_targets, val_preds)
    print(f"Final Validation Metric: {final_log_loss:.15f}")

    # 5. Failure Analysis
    print("\n--- Failure Analysis ---")
    errors = np.abs(val_preds - val_targets)

    # Correlation with Incidence Angle
    # Handle NaNs in inc_angle if any (though loader imputes them, let's be safe)
    valid_mask = ~np.isnan(val_inc)
    if np.sum(valid_mask) > 1:
        corr_inc, _ = pearsonr(errors[valid_mask], val_inc[valid_mask])
        print(f"Correlation (Error vs Inc Angle): {corr_inc:.4f}")
    else:
        print("Correlation (Error vs Inc Angle): N/A (Insufficient data)")

    # Correlation with Image Intensity
    corr_int, _ = pearsonr(errors, val_img_means)
    print(f"Correlation (Error vs Mean Image Intensity): {corr_int:.4f}")

    # 6. Submission
    THRESHOLD = 0.17493283735739185

    if final_log_loss < THRESHOLD:
        print(f"\nMetric {final_log_loss:.6f} < {THRESHOLD}. Generating submission...")

        test_loader = get_loader(
            df_test,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            augment=False,
            load_cached_data=True,
        )

        test_preds, _, _, _ = predict_ensemble(test_loader, models, device)

        submission_df = pd.DataFrame({"id": df_test["id"], "is_iceberg": test_preds})

        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
    else:
        print(f"\nMetric {final_log_loss:.6f} >= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()

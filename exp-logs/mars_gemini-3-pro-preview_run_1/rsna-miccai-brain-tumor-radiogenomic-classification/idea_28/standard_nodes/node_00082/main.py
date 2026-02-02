import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import from provided library
from library.config import Config, setup_system
from library.utils import get_device, calculate_roc_auc, save_submission
from library.train import run_training_fold
from library.data import get_dataloader
from library.model import SIRVEfficientNet


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Ensure fast baseline execution by limiting epochs if necessary,
    # though 20 is generally fast enough on this dataset size.
    # We will stick to the config default of 20 to ensure convergence.
    setup_system()
    device = get_device()

    # ==========================================
    # 2. Integrity Verification
    # ==========================================
    print("Verifying dataset integrity...")
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Combine and filter exclusions to check total count
    df_full = pd.concat([df_train, df_val])
    df_full = df_full[~df_full["BraTS21ID"].isin(Config.EXCLUDE_CASES)]

    total_subjects = len(df_full)
    print(f"Total subjects available for training: {total_subjects}")

    # Expectation is around 523 subjects
    if total_subjects < 500:
        raise RuntimeError(
            f"Dataset integrity check failed. Found {total_subjects} subjects, expected ~523."
        )

    # ==========================================
    # 3. Training Loop (5-Fold CV)
    # ==========================================
    print("Starting 5-Fold Cross-Validation...")
    fold_aucs = []

    for fold in range(Config.N_FOLDS):
        auc = run_training_fold(fold)
        fold_aucs.append(auc)

    print(f"Training complete. Fold AUCs: {fold_aucs}")

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    print("Generating OOF predictions for global validation metric...")

    oof_ids = []
    oof_preds = []
    oof_targets = []

    # Iterate through folds to generate predictions on the validation splits
    for fold in range(Config.N_FOLDS):
        # Load the validation loader for this fold
        val_loader = get_dataloader(
            split="val",
            fold_idx=fold,
            batch_size=Config.BATCH_SIZE,
            load_cached_data=True,
        )

        # Load the best model for this fold
        model = SIRVEfficientNet(num_classes=1)
        model_path = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        # Get IDs from the dataset (order is preserved in val_loader)
        # Note: val_loader.dataset is the subset for validation
        fold_ids = val_loader.dataset.df["BraTS21ID"].values

        fold_preds_list = []
        fold_targets_list = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)

                # Forward pass
                logits = model(images)
                probs = torch.sigmoid(logits).cpu().numpy().flatten()

                fold_preds_list.extend(probs)
                fold_targets_list.extend(targets.numpy().flatten())

        # Append to global OOF lists
        oof_ids.extend(fold_ids)
        oof_preds.extend(fold_preds_list)
        oof_targets.extend(fold_targets_list)

    # Convert to arrays
    oof_preds = np.array(oof_preds)
    oof_targets = np.array(oof_targets)

    # Compute Final Metric
    final_metric = calculate_roc_auc(oof_targets, oof_preds)
    print(f"Final Validation Metric: {final_metric}")

    # Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(oof_preds - oof_targets)

    # Load cache to get slice counts (metadata features)
    cache_path = os.path.join(Config.CACHE_DIR, "roi_cache_train_val.parquet")
    if os.path.exists(cache_path):
        cache_df = pd.read_parquet(cache_path)

        # Calculate slice count per modality for each subject
        cache_df["slice_count"] = cache_df["Files"].apply(
            lambda x: len(x.split(";")) if x else 0
        )

        # Pivot to get features: BraTS21ID | FLAIR_count | T1wCE_count | T2w_count
        features_df = cache_df.pivot(
            index="BraTS21ID", columns="Modality", values="slice_count"
        ).reset_index()
        features_df.columns = ["BraTS21ID", "FLAIR_count", "T1wCE_count", "T2w_count"]

        # Merge errors with features
        analysis_df = pd.DataFrame({"BraTS21ID": oof_ids, "error": errors})
        analysis_df = analysis_df.merge(features_df, on="BraTS21ID", how="left").fillna(
            0
        )

        print("Correlation between Error Magnitude and Slice Counts:")
        for col in ["FLAIR_count", "T1wCE_count", "T2w_count"]:
            if col in analysis_df.columns:
                corr, _ = pearsonr(analysis_df["error"], analysis_df[col])
                print(f"  {col}: {corr:.4f}")
    else:
        print("Warning: ROI cache not found. Skipping detailed correlation analysis.")

    # ==========================================
    # 5. Submission
    # ==========================================
    THRESHOLD = 0.6705454545454544

    if final_metric > THRESHOLD:
        print(f"\nMetric {final_metric} > {THRESHOLD}. Generating submission...")

        # Load Test Loader
        test_loader = get_dataloader(
            split="test", batch_size=Config.BATCH_SIZE, load_cached_data=True
        )
        test_ids = test_loader.dataset.df["BraTS21ID"].values

        # Ensemble Prediction
        avg_preds = np.zeros(len(test_ids))

        for fold in range(Config.N_FOLDS):
            print(f"Inference with model fold {fold}...")
            model = SIRVEfficientNet(num_classes=1)
            model_path = os.path.join(Config.CACHE_DIR, f"best_model_fold{fold}.pth")
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            fold_preds = []
            with torch.no_grad():
                for images, _ in test_loader:
                    images = images.to(device)
                    logits = model(images)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    fold_preds.extend(probs)

            avg_preds += np.array(fold_preds)

        # Average
        avg_preds /= Config.N_FOLDS

        # Save
        save_submission(test_ids, avg_preds, Config.SUBMISSION_PATH)
    else:
        print(f"\nMetric {final_metric} <= {THRESHOLD}. Submission skipped.")


if __name__ == "__main__":
    main()

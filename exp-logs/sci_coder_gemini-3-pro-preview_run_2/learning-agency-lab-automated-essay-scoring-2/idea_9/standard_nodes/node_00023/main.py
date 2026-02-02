import os
import sys
import shutil
import pandas as pd
import numpy as np
import torch

# Import library modules
from library.config import Config
from library.utils import seed_everything, compute_qwk, get_logger
from library.models_semantic import run_semantic_training, predict_semantic_test
from library.models_classic import run_classic_branches
from library.stacking import StackingModel, optimize_thresholds
from library.features import extract_linguistic_features


def create_consistent_subsets(n_samples=6000):
    """
    Creates consistent subsets of train and val data to ensure
    all branches use the same data and the pipeline runs fast.
    """
    print(f"Creating data subsets (Target Total: {n_samples})...")

    # Load original metadata
    df_train = pd.read_csv("./metadata/train.csv")
    df_val = pd.read_csv("./metadata/val.csv")

    # Calculate proportions
    total_original = len(df_train) + len(df_val)
    frac = min(1.0, n_samples / total_original)

    # Stratified subsample
    # We use the 'score' column for stratification
    # We sample train and val separately to maintain the split structure

    # Subsample Train
    df_train_sub = (
        df_train.groupby("score", group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=Config.seed))
        .reset_index(drop=True)
    )

    # Subsample Val
    df_val_sub = (
        df_val.groupby("score", group_keys=False)
        .apply(lambda x: x.sample(frac=frac, random_state=Config.seed))
        .reset_index(drop=True)
    )

    print(f"Subset Train Shape: {df_train_sub.shape}")
    print(f"Subset Val Shape: {df_val_sub.shape}")

    # Save to working directory
    os.makedirs(Config.working_dir, exist_ok=True)
    train_sub_path = os.path.join(Config.working_dir, "train_subset.csv")
    val_sub_path = os.path.join(Config.working_dir, "val_subset.csv")

    df_train_sub.to_csv(train_sub_path, index=False)
    df_val_sub.to_csv(val_sub_path, index=False)

    return train_sub_path, val_sub_path


def main():
    # --- 1. Setup & Configuration ---
    seed_everything(Config.seed)

    # Clean output directory to prevent stale cache issues from previous runs
    if os.path.exists(Config.output_dir):
        shutil.rmtree(Config.output_dir)
    os.makedirs(Config.output_dir, exist_ok=True)

    # Create subsets for speed
    train_path, val_path = create_consistent_subsets(n_samples=6000)

    # Override Config
    Config.train_path = train_path
    Config.val_path = val_path
    Config.debug = False  # We handle subsetting manually via file paths
    Config.epochs = 1  # Fast baseline
    Config.use_awp = False  # Disable for speed
    Config.train_batch_size = 4  # Safe for A100 with DeBERTa-Large
    Config.n_folds = 5  # Standard CV

    # Initialize environment
    Config.setup()
    logger = get_logger("pipeline")
    logger.info("Configuration configured for fast baseline run.")

    # --- 2. Semantic Branch (DeBERTa) ---
    logger.info(">>> Starting Semantic Branch...")
    run_semantic_training()
    predict_semantic_test()

    # --- 3. Classic Branches (Ridge) ---
    logger.info(">>> Starting Classic Branches...")
    run_classic_branches()

    # --- 4. Stacking (Meta-Learner) ---
    logger.info(">>> Starting Stacking Model...")
    stacker = StackingModel()
    # Train returns the OOF predictions for the meta-learner and the true labels
    oof_preds, y_true = stacker.train()

    # --- 5. Validation & Failure Analysis ---
    val_qwk = compute_qwk(y_true, oof_preds)
    print(f"Final Validation Metric: {val_qwk}")

    logger.info(">>> Performing Failure Analysis...")
    # Load feature data for correlation analysis
    # We need the merged train+val dataframe corresponding to our subset
    df_train = pd.read_csv(Config.train_path)
    df_val = pd.read_csv(Config.val_path)
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

    # Extract/Load features (Stacking step already cached them)
    features_df = extract_linguistic_features(
        df_full, split="train_val_merged", load_cached_data=True
    )

    # Calculate absolute error
    errors = np.abs(y_true - oof_preds)

    # Calculate correlations
    correlations = {}
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        # Handle potential NaN or constant columns
        if features_df[col].std() > 0:
            corr = np.corrcoef(errors, features_df[col])[0, 1]
            correlations[col] = corr
        else:
            correlations[col] = 0.0

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, val in sorted_corr[:5]:
        print(f"{name}: {val:.4f}")

    # --- 6. Submission ---
    TARGET_METRIC = 0.8307992749024942

    if val_qwk > TARGET_METRIC:
        logger.info(f"Metric {val_qwk:.6f} > {TARGET_METRIC}. Generating submission...")

        # Optimize Thresholds
        best_thresholds = optimize_thresholds(y_true, oof_preds)

        # Predict on Test Set
        test_preds_continuous, essay_ids = stacker.predict()

        # Apply Thresholds
        # np.digitize returns indices 0..N. We map 0->1, 1->2, etc.
        test_preds_int = np.digitize(test_preds_continuous, best_thresholds) + 1
        test_preds_int = np.clip(test_preds_int, 1, 6)

        # Save Submission
        submission_df = pd.DataFrame({"essay_id": essay_ids, "score": test_preds_int})

        os.makedirs(Config.submission_dir, exist_ok=True)
        submission_df.to_csv(Config.submission_path, index=False)
        print(f"Submission saved to {Config.submission_path}")
        print(submission_df.head())

    else:
        logger.warning(f"Metric {val_qwk:.6f} <= {TARGET_METRIC}. Submission skipped.")


if __name__ == "__main__":
    main()

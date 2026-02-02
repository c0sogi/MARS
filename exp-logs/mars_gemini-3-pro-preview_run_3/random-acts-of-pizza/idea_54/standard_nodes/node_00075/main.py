import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

# Import provided library modules
from library.config import SEED, TARGET_COL, ALLOW_LIST_METADATA, CACHE_DIR
from library.data_loader import load_raw_dataset, clean_dataset, get_stratified_folds
from library.feature_engineering import FeatureFactory
from library.training_engine import HybridTrainer
from library.inference_engine import HybridPredictor


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def cleanup_test_cache():
    """
    Removes cached files associated with the 'test' split.
    This is necessary because HybridPredictor hardcodes the cache key to 'test',
    so we must clear it between validation (which uses predict) and actual test submission.
    """
    pattern = os.path.join(CACHE_DIR, "X_test_*")
    for f in glob.glob(pattern):
        try:
            os.remove(f)
        except OSError:
            pass


def main():
    # 1. Setup
    set_seeds(SEED)
    warnings.filterwarnings("ignore")

    # 2. Data Loading
    # We load 'train' for training and 'val' for the hold-out validation metric.
    # We do not use 'full_train' to ensure a strict validation score is computed.
    train_df = load_raw_dataset("train")
    val_df = load_raw_dataset("val")
    test_df = load_raw_dataset("test")

    # Clean datasets
    # train and val have targets, test does not
    train_df = clean_dataset(train_df, is_test=False)
    val_df = clean_dataset(val_df, is_test=False)
    test_df = clean_dataset(test_df, is_test=True)

    # 3. Feature Engineering
    # Fit only on training data to avoid leakage
    print("Fitting FeatureFactory on training data...")
    feature_factory = FeatureFactory()
    feature_factory.fit(train_df)

    # 4. Training
    print("Starting Hybrid Training...")
    trainer = HybridTrainer(feature_factory)

    # Generate stratified folds for the training set
    folds = get_stratified_folds(train_df)

    # Train Level 1: Base Learners
    # This generates OOF predictions on train_df and saves models
    oof_preds = trainer.train_level_1(train_df, folds)

    # Train Level 2: Meta-Learner
    # Trains on the OOF predictions
    y_train = train_df[TARGET_COL].values
    trainer.train_level_2(oof_preds, y_train)

    # 5. Validation Assessment
    print("\n=== Starting Validation Assessment ===")
    predictor = HybridPredictor(feature_factory)

    # Ensure cache is clean before validation inference
    cleanup_test_cache()

    # Predict on Validation Set
    # Note: predictor.predict uses cache key "test" internally
    val_probs = predictor.predict(val_df)
    y_val = val_df[TARGET_COL].values

    # Compute Metric
    val_auc = roc_auc_score(y_val, val_probs)
    print(f"Final Validation Metric: {val_auc}")

    # 6. Failure Analysis
    print("\n=== Failure Analysis ===")
    errors = np.abs(y_val - val_probs)

    # Retrieve the metadata features used for validation (cached as "test")
    # We use these for correlation analysis
    val_features = feature_factory.transform(val_df, "test", load_cache=True)
    X_meta_val = val_features["metadata"]

    # Calculate correlation between Error and Metadata features
    meta_cols = ALLOW_LIST_METADATA
    correlations = []

    if X_meta_val.shape[1] == len(meta_cols):
        for i, col_name in enumerate(meta_cols):
            feat_values = X_meta_val[:, i]
            # Avoid division by zero if feature is constant
            if np.std(feat_values) > 1e-9:
                corr = np.corrcoef(feat_values, errors)[0, 1]
                correlations.append((col_name, corr))
            else:
                correlations.append((col_name, 0.0))

        # Sort by absolute correlation
        correlations.sort(key=lambda x: abs(x[1]), reverse=True)

        print("Correlation between Model Error and Input Features:")
        for col, corr in correlations:
            print(f"  {col}: {corr:.4f}")
    else:
        print("Skipping detailed metadata correlation due to shape mismatch.")

    # 7. Submission
    # Clean cache again so we don't use validation features for test prediction
    cleanup_test_cache()

    threshold = 0.7222984867326668
    if val_auc > threshold:
        print(f"\nValidation metric {val_auc} > {threshold}. Generating submission...")
        predictor.generate_submission(test_df)
    else:
        print(f"\nValidation metric {val_auc} <= {threshold}. Submission skipped.")


if __name__ == "__main__":
    main()

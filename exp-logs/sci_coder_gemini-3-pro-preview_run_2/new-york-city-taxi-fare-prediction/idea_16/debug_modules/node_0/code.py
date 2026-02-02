import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Import library modules
from library.config import Config
from library.data_processor import prepare_datasets, load_data
from library.margin_logic import construct_train_margins, construct_test_margins
from library.feature_generator import process_features, prepare_dmatrix
from library.model_wrapper import XGBResiduaLearner

# Ensure reproducibility
np.random.seed(42)

# ---------------------------------------------------------
# Configuration Overrides for Demonstration
# ---------------------------------------------------------
# We modify the Config class attributes directly to optimize for speed
# and to ensure we use a clean working directory for this run.
Config.TRAIN_SUBSAMPLE_SIZE = 100_000  # Train on a smaller subset for speed
Config.NUM_BOOST_ROUND = 100  # Reduce boosting rounds
Config.EARLY_STOPPING_ROUNDS = 10  # Stop early if no improvement
Config.WORKING_DIR = "./working/demo_execution"  # Dedicated demo directory
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Create the working directory
os.makedirs(Config.WORKING_DIR, exist_ok=True)


def main():
    print("=== Starting Hierarchical Residual Dual-Hygiene Pipeline Demo ===")

    # ---------------------------------------------------------
    # 1. Data Preparation
    # ---------------------------------------------------------
    # Loads the full training data, applies strict filtering to create the "Wisdom Set"
    # (used for statistics), and loose filtering+subsampling for the "Learner Set".
    print("\n[Step 1] Preparing Datasets...")
    wisdom_df, learner_df = prepare_datasets(load_cached_data=False)

    print(f"   Wisdom Set (Statistics) Size: {len(wisdom_df):,}")
    print(f"   Learner Set (Training) Size:  {len(learner_df):,}")

    # Validation
    assert len(wisdom_df) > 0, "Wisdom set should not be empty."
    assert (
        len(learner_df) <= Config.TRAIN_SUBSAMPLE_SIZE
    ), "Learner set size exceeds limit."

    # ---------------------------------------------------------
    # 2. Margin Construction
    # ---------------------------------------------------------
    # This is the core of the strategy. We calculate a 'base_margin' (expected price)
    # based on hierarchical geohash statistics.
    print("\n[Step 2] Constructing Margins...")

    # A. Train Margins: Uses K-Fold Leave-One-Out subtraction to prevent data leakage.
    # The stats used for a specific fold are calculated from the Wisdom set MINUS that fold.
    print("   Constructing Training Margins (K-Fold Subtraction)...")
    learner_with_margins = construct_train_margins(
        learner_df, wisdom_df, load_cached_data=False
    )

    # Validate Train Margins
    assert (
        "margin" in learner_with_margins.columns
    ), "Margin column missing in training set."
    assert (
        learner_with_margins["margin"].notna().all()
    ), "NaNs found in training margins."

    # B. Validation Margins: Uses the full Wisdom set statistics.
    # We load a subset of the validation data for evaluation.
    print("   Constructing Validation Margins...")
    val_full = load_data(Config.VAL_DATA_PATH)
    val_df = val_full.sample(n=20_000, random_state=Config.SEED).copy()
    val_with_margins = construct_test_margins(val_df, wisdom_df, load_cached_data=False)

    # Validate Validation Margins
    assert (
        "margin" in val_with_margins.columns
    ), "Margin column missing in validation set."

    # ---------------------------------------------------------
    # 3. Feature Engineering
    # ---------------------------------------------------------
    print("\n[Step 3] Feature Engineering...")

    # Process features (temporal, geometric, etc.)
    train_feat = process_features(
        learner_with_margins, cache_key="processed_train", load_cached_data=False
    )
    val_feat = process_features(
        val_with_margins, cache_key="processed_val", load_cached_data=False
    )

    # Define the list of features to use for training
    exclude_cols = {"key", "fare_amount", "margin", "pickup_datetime"}
    feature_cols = [c for c in train_feat.columns if c not in exclude_cols]
    print(f"   Using {len(feature_cols)} features: {feature_cols}")

    # ---------------------------------------------------------
    # 4. Model Training
    # ---------------------------------------------------------
    print("\n[Step 4] Training XGBoost Model (Residual Learning)...")

    # Create DMatrix objects. Crucially, we pass 'base_margin'.
    # The model will learn: Target ~ Base_Margin + f(Features)
    dtrain = prepare_dmatrix(train_feat, feature_cols, target_col="fare_amount")
    dval = prepare_dmatrix(val_feat, feature_cols, target_col="fare_amount")

    # Initialize and Train
    learner = XGBResiduaLearner()
    learner.train(dtrain, dval)

    # Save the model
    learner.save("xgb_model.json")

    # ---------------------------------------------------------
    # 5. Evaluation
    # ---------------------------------------------------------
    print("\n[Step 5] Evaluating Model...")

    # Predict on validation set
    val_preds = learner.predict(dval)

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(val_feat["fare_amount"], val_preds))
    print(f"   Validation RMSE: {rmse:.4f}")

    # Sanity check: RMSE should be reasonable (e.g., < $10.00 for a good model, < $15.00 for a basic one)
    if rmse > 15.0:
        print("   WARNING: RMSE is high. Check feature engineering or margin logic.")
    else:
        print("   RMSE is within expected range.")

    # ---------------------------------------------------------
    # 6. Submission Generation
    # ---------------------------------------------------------
    print("\n[Step 6] Generating Submission for Test Set...")

    # Load actual test set
    test_df = load_data(Config.TEST_DATA_PATH)

    # Apply the pipeline to the test set
    # 1. Construct Margins (using full Wisdom stats)
    test_with_margins = construct_test_margins(
        test_df, wisdom_df, load_cached_data=False
    )

    # 2. Generate Features
    test_feat = process_features(
        test_with_margins, cache_key="processed_test", load_cached_data=False
    )

    # 3. Create DMatrix (with margin)
    dtest = prepare_dmatrix(test_feat, feature_cols)

    # 4. Predict
    test_preds = learner.predict(dtest)

    # 5. Save Submission
    learner.generate_submission(test_df, test_preds)

    print("\n=== Demo Completed Successfully ===")
    print(f"Submission saved to: {Config.SUBMISSION_PATH}")


if __name__ == "__main__":
    main()

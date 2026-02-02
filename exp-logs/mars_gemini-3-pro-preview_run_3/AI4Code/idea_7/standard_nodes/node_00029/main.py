import os
import sys
import numpy as np
import pandas as pd
import warnings

# Import library modules
from library.config import Config
from library.data_utils import set_seed, get_data_splits
from library.fine_tuning import run_fine_tuning
from library.feature_engineering import generate_features
from library.lgbm_model import (
    train_regressor,
    validate_model,
    generate_submission,
    predict_rank,
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration & Setup
    # -------------------------------------------------------------------------
    # Adjust Config for a fast but effective baseline run
    # We use a larger debug sample size (15k) to ensure good performance
    # while staying well within the 2-hour runtime limit on A100.
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 15000
    Config.EPOCHS = 1

    # Set seeds for reproducibility
    set_seed(Config.SEED)

    print(f"Starting HC-SSR Pipeline")
    print(
        f"Configuration: DEBUG={Config.DEBUG}, SAMPLE_SIZE={Config.DEBUG_SAMPLE_SIZE}, EPOCHS={Config.EPOCHS}"
    )

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Clear specific feature caches to ensure we use the fresh model features
    # This guarantees the "start to finish" integrity of the run
    for path in [
        Config.TRAIN_FEATURES_PATH,
        Config.VAL_FEATURES_PATH,
        Config.TEST_FEATURES_PATH,
    ]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass

    # -------------------------------------------------------------------------
    # 2. Semantic Backbone Fine-Tuning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Stage 1: Semantic Backbone Fine-Tuning")
    print("=" * 40)
    # This will load the sampled training data and fine-tune the MPNet model
    run_fine_tuning()

    # -------------------------------------------------------------------------
    # 3. Feature Engineering (Train/Val)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Stage 2: Feature Engineering")
    print("=" * 40)

    # Load data splits (respects Config.DEBUG)
    df_train, df_val, _ = get_data_splits()

    # Generate features using the fine-tuned model
    # load_cached_data=False forces regeneration using the new model
    train_features = generate_features(df_train, mode="train", load_cached_data=False)
    val_features = generate_features(df_val, mode="val", load_cached_data=False)

    # -------------------------------------------------------------------------
    # 4. LightGBM Regression Training
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Stage 3: LightGBM Training")
    print("=" * 40)

    lgbm_model = train_regressor(train_features, val_features)

    # -------------------------------------------------------------------------
    # 5. Validation & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Stage 4: Validation & Failure Analysis")
    print("=" * 40)

    # Compute and print the official metric
    final_metric = validate_model(lgbm_model, df_val, val_features)
    print(f"Final Validation Metric: {final_metric:.16f}")

    # Failure Analysis
    print("\nPerforming Failure Analysis...")
    if not val_features.empty:
        # Predict on validation set to get errors
        val_preds = predict_rank(lgbm_model, val_features)

        analysis_df = val_features.copy()
        analysis_df["pred"] = val_preds
        analysis_df["error"] = np.abs(analysis_df["target"] - analysis_df["pred"])

        # Calculate correlations between error and features
        corr_cols = [
            "error",
            "n_code",
            "md_len",
            "sim_max",
            "best_match_loc",
            "center_of_mass",
        ]
        # Filter columns that exist in the dataframe
        existing_cols = [c for c in corr_cols if c in analysis_df.columns]

        if "error" in existing_cols:
            correlations = (
                analysis_df[existing_cols].corr()["error"].sort_values(ascending=False)
            )
            print("Correlation between Error Magnitude and Features:")
            print(correlations)
        else:
            print("Could not calculate correlations: missing columns.")
    else:
        print("Validation features empty, skipping failure analysis.")

    # -------------------------------------------------------------------------
    # 6. Submission Generation
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print(" Stage 5: Submission")
    print("=" * 40)

    threshold = 0.8061

    if final_metric > threshold:
        print(
            f"Metric ({final_metric:.4f}) > Threshold ({threshold}). Proceeding with submission."
        )

        # IMPORTANT: We must predict on the FULL test set, not the debug sample.
        # We temporarily disable DEBUG to load the full test metadata.
        print("Loading full test set for inference...")
        Config.DEBUG = False
        _, _, df_test_full = get_data_splits()

        # Generate features for the full test set
        test_features = generate_features(
            df_test_full, mode="test", load_cached_data=False
        )

        # Generate submission file
        generate_submission(df_test_full, test_features, lgbm_model)
    else:
        print(
            f"Metric ({final_metric:.4f}) <= Threshold ({threshold}). Submission skipped."
        )


if __name__ == "__main__":
    main()

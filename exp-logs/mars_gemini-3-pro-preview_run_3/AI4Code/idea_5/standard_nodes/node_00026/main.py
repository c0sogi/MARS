import os
import sys
import pandas as pd
import numpy as np
import torch
import warnings

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.fine_tune import fine_tune_models
from library.feature_engineering import generate_features
from library.regressor import train_lgbm, generate_submission

# Suppress warnings
warnings.filterwarnings("ignore")


def perform_failure_analysis(val_df, val_preds_df, val_metadata):
    """
    Analyzes model performance on the validation set.
    Correlates absolute error with features to identify error sources.
    """
    print("\nPerforming Failure Analysis...")

    # Merge predictions with targets and features
    # val_df contains 'target' and features
    # val_preds_df contains 'pred_rank' (which is normalized rank * n_code? No, predict_ranks returns raw prediction from model)
    # Actually, predict_ranks returns the raw model output. In this pipeline, the target is normalized rank [0,1].
    # So the model predicts normalized rank.

    # We need to join the predictions back to the feature dataframe to get the error
    # val_df has 'id' (cell_id), 'notebook_id', 'target', and features

    # predict_ranks returns dataframe with ['id', 'notebook_id', 'pred_rank']
    # But wait, predict_ranks in library/regressor.py returns whatever model.predict returns.
    # The model is trained on 'target' which is normalized rank.

    analysis_df = val_df.merge(val_preds_df[["id", "pred_rank"]], on="id", how="left")
    analysis_df["abs_error"] = (analysis_df["target"] - analysis_df["pred_rank"]).abs()

    # Select numerical features for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
    # Exclude non-feature columns
    exclude = ["target", "pred_rank", "abs_error"]
    feature_cols = [c for c in numeric_cols if c not in exclude]

    correlations = {}
    for col in feature_cols:
        correlations[col] = analysis_df["abs_error"].corr(analysis_df[col])

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Absolute Error:")
    for name, corr in sorted_corr[:5]:
        print(f"  {name}: {corr:.4f}")

    return analysis_df


def main():
    # 1. Configuration and Setup
    set_seed(Config.SEED)

    # Configure for fast baseline execution
    # We use a subset for training to meet time constraints, but full validation/test for accurate metrics
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 10000

    print(
        f"Starting execution with DEBUG={Config.DEBUG}, SAMPLE_SIZE={Config.DEBUG_SAMPLE_SIZE}"
    )

    # 2. Fine-Tune Models (Contrastive Learning)
    # This uses Config.DEBUG internally to limit the dataset size
    print("\n--- Step 1: Fine-Tuning Models ---")
    fine_tune_models(load_cached_data=True)

    # 3. Feature Engineering
    print("\n--- Step 2: Feature Engineering ---")

    # Generate features for Training set (Subset)
    # We pass debug=True to use the sample size defined in Config
    print("Generating Training Features...")
    train_features = generate_features(
        Config.TRAIN_METADATA_PATH,
        Config.TRAIN_FEATURES_PATH,
        load_cached_data=True,
        debug=True,
    )

    # Generate features for Validation set (Full)
    # We must evaluate on the full validation set as per requirements
    print("Generating Validation Features...")
    val_features = generate_features(
        Config.VAL_METADATA_PATH,
        Config.VAL_FEATURES_PATH,
        load_cached_data=True,
        debug=False,
    )

    # 4. Train Regressor
    print("\n--- Step 3: Training Regressor ---")
    model = train_lgbm(train_features, val_features)

    # 5. Validation and Metrics
    print("\n--- Step 4: Validation & Analysis ---")

    # Reconstruct order for validation set to compute Kendall Tau
    # We use generate_submission logic but point it to validation metadata
    # Note: generate_submission saves to file, but also returns the DF.
    # We will temporarily save a val submission file.
    temp_val_sub_path = os.path.join(Config.WORKING_DIR, "temp_val_submission.csv")

    # Temporarily override submission path in Config to reuse the function without modifying library
    original_sub_path = Config.SUBMISSION_PATH
    Config.SUBMISSION_PATH = temp_val_sub_path

    val_pred_df = generate_submission(model, val_features, Config.VAL_METADATA_PATH)

    # Restore config
    Config.SUBMISSION_PATH = original_sub_path

    # Load Ground Truth
    val_gt_df = pd.read_csv(Config.VAL_METADATA_PATH)

    # Compute Metric
    kt_score = compute_kendall_tau(val_pred_df, val_gt_df)
    print(f"Final Validation Metric: {kt_score}")

    # Failure Analysis
    # We need raw predictions for correlation analysis
    from library.regressor import predict_ranks

    raw_val_preds = predict_ranks(model, val_features)
    perform_failure_analysis(val_features, raw_val_preds, val_gt_df)

    # 6. Submission
    print("\n--- Step 5: Submission ---")
    THRESHOLD = 0.8061

    if kt_score > THRESHOLD:
        print(f"Validation score ({kt_score}) > {THRESHOLD}. Generating submission...")

        # Generate features for Test set (Full)
        print("Generating Test Features...")
        test_features = generate_features(
            Config.TEST_METADATA_PATH,
            Config.TEST_FEATURES_PATH,
            load_cached_data=True,
            debug=False,
        )

        generate_submission(model, test_features, Config.TEST_METADATA_PATH)
        print("Submission generation complete.")
    else:
        print(
            f"Validation score ({kt_score}) <= {THRESHOLD}. Skipping submission generation."
        )


if __name__ == "__main__":
    main()

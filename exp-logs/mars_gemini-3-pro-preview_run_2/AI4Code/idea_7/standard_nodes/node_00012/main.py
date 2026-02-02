import pandas as pd
import numpy as np
import os
import sys

# Import from provided libraries
from library.config import Config
from library.modeling import StackedRanker
from library.metrics import score_dataframe


def main():
    # 1. Setup Environment
    Config.setup()

    # 2. Initialize the Stacked Ranker
    ranker = StackedRanker()

    # 3. Feature Engineering (Train & Validation)
    print("Step 1: Feature Engineering...")
    # Process splits (handles caching automatically)
    df_train, df_nb_train = ranker.fe.process_split("train")
    df_val, df_nb_val = ranker.fe.process_split("val")

    # Load the fitted vectorization pipeline (fit during train processing)
    ranker.pipeline.load(ranker.models_dir)

    # 4. Train Stage 1: Sparse Lexical Regressor (Ridge)
    print("Step 2: Training Stage 1 (Ridge)...")
    train_oof, val_preds_stage1, ridge_model = ranker.train_stage1(df_train, df_val)

    # 5. Train Stage 2: Anchor-Aware Gradient Booster (LightGBM)
    print("Step 3: Training Stage 2 (LightGBM)...")
    lgbm_model = ranker.train_stage2(df_train, df_val, train_oof, val_preds_stage1)

    # 6. Validation Inference & Scoring
    print("Step 4: Validation Scoring...")

    # Prepare stacked features for validation set
    X_val_stk = ranker._prepare_stage2_features(df_val, val_preds_stage1)

    # Generate final rank predictions
    val_final_preds = lgbm_model.predict(X_val_stk)

    # Construct cell orders (merging predicted markdown ranks with fixed code ranks)
    val_submission_df = ranker._generate_submission(df_val, df_nb_val, val_final_preds)

    # Convert to dictionary for scoring
    val_predictions = dict(
        zip(val_submission_df["id"], val_submission_df["cell_order"])
    )

    # Load Ground Truth Metadata
    val_metadata = pd.read_csv(Config.VAL_METADATA_PATH)

    # Calculate Kendall Tau Metric
    final_metric = score_dataframe(val_metadata, val_predictions)
    print(f"Final Validation Metric: {final_metric}")

    # 7. Failure Analysis
    print("Step 5: Failure Analysis...")

    # Create analysis dataframe
    df_analysis = df_val.copy()
    df_analysis["pred_rank"] = val_final_preds
    # Calculate Absolute Rank Error
    df_analysis["rank_error"] = np.abs(df_analysis["rank"] - df_analysis["pred_rank"])
    # Add Stage 1 predictions for correlation check
    df_analysis["ridge_pred"] = val_preds_stage1

    # Define features to correlate with error
    features_to_corr = [
        "anchor_sim",
        "top3_anchor_rank_mean",
        "total_cells",
        "md_ratio",
        "ridge_pred",
    ]

    print("Correlation between Absolute Rank Error and Features:")
    for feat in features_to_corr:
        if feat in df_analysis.columns:
            corr_val = df_analysis["rank_error"].corr(df_analysis[feat])
            print(f"  {feat}: {corr_val:.8f}")

    # 8. Submission Generation
    THRESHOLD = 0.7623647875869406

    if final_metric > THRESHOLD:
        print(
            f"\nMetric ({final_metric}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        # Process Test Data
        df_test, df_nb_test = ranker.fe.process_split("test")

        # Stage 1 Inference (Ridge)
        print("Generating Stage 1 Test Predictions...")
        X_test = ranker._get_tfidf_matrix(df_test)
        test_ridge_preds = ridge_model.predict(X_test)

        # Stage 2 Inference (LightGBM)
        print("Generating Stage 2 Test Predictions...")
        X_test_stk = ranker._prepare_stage2_features(df_test, test_ridge_preds)
        test_final_preds = lgbm_model.predict(X_test_stk)

        # Generate Submission File
        print("Constructing final cell orders...")
        sub_df = ranker._generate_submission(df_test, df_nb_test, test_final_preds)

        # Save to disk
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nMetric ({final_metric}) <= Threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()

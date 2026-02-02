import os
import sys
import numpy as np
import pandas as pd
import random
import torch

# Import from the provided library files
from library.config import Config
from library.pipeline import RankingPipeline


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------
def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def calculate_correlation(x, y):
    """Calculates Pearson correlation using numpy."""
    # valid indices where neither x nor y is nan
    valid = np.isfinite(x) & np.isfinite(y)
    if not valid.any():
        return 0.0
    return np.corrcoef(x[valid], y[valid])[0, 1]


# ------------------------------------------------------------------------------
# Main Execution
# ------------------------------------------------------------------------------
def main():
    # 1. Setup
    set_seed(Config.SEED)
    print("=== AI Notebook Ordering Pipeline ===")

    # 2. Initialize Pipeline
    # The RankingPipeline class orchestrates the entire workflow defined in the library
    pipeline = RankingPipeline()

    # 3. Train Pipeline
    # This fits the TF-IDF/SVD, trains the Ridge (Stage 1), extracts features,
    # and trains the LightGBM (Stage 2).
    # We enable caching to speed up re-runs if data exists.
    print("\n[Step 1] Executing Training Pipeline...")
    pipeline.train(load_cached_data=True)

    # 4. Validation Assessment
    # We manually run inference on the validation set to get the exact metric and data for analysis.
    print("\n[Step 2] Assessing Validation Performance...")

    # Load validation data and features
    df_val = pipeline.loader.load_val_data(load_cached_data=True)
    val_features = pipeline.extractor.extract_features(
        df_val, pipeline.text_pipeline, mode="val", load_cached_data=True
    )

    # Generate predictions
    val_s1_preds = pipeline.stage1.predict(df_val, pipeline.text_pipeline)
    val_final_preds = pipeline.stage2.predict(val_features, val_s1_preds)

    # Calculate Metric
    val_metric = pipeline._evaluate_kendall(df_val, val_final_preds)

    # REQUIRED: Print the final validation metric in the specific format
    print(f"Final Validation Metric: {val_metric}")

    # 5. Failure Analysis
    print("\n[Step 3] Performing Failure Analysis...")

    # Merge features with predictions and targets
    # val_features contains 'target_rank'
    # val_final_preds contains 'pred_rank'
    analysis_df = val_features.merge(
        val_final_preds[["id", "cell_id", "pred_rank"]],
        on=["id", "cell_id"],
        how="inner",
    )

    # Calculate absolute error
    analysis_df["error"] = np.abs(analysis_df["target_rank"] - analysis_df["pred_rank"])

    # Select feature columns (exclude IDs and targets)
    exclude_cols = {"id", "cell_id", "target_rank", "pred_rank", "error"}
    feature_cols = [c for c in analysis_df.columns if c not in exclude_cols]

    # Calculate correlations between features and error
    correlations = []
    for col in feature_cols:
        # Check if column is numeric
        if pd.api.types.is_numeric_dtype(analysis_df[col]):
            # Skip constant columns
            if analysis_df[col].std() > 1e-9:
                corr = calculate_correlation(analysis_df["error"], analysis_df[col])
                correlations.append((col, corr))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Features correlated with Error Magnitude:")
    for name, corr in correlations[:10]:
        print(f"  {name}: {corr:.4f}")

    # 6. Submission Logic
    THRESHOLD = 0.7959051868218839

    if val_metric > THRESHOLD:
        print(
            f"\n[Step 4] Metric ({val_metric}) > Threshold ({THRESHOLD}). Generating Submission..."
        )
        # pipeline.predict handles test loading, feature extraction, inference, and CSV generation
        pipeline.predict(load_cached_data=True)
    else:
        print(
            f"\n[Step 4] Metric ({val_metric}) <= Threshold ({THRESHOLD}). Submission Skipped."
        )


if __name__ == "__main__":
    main()

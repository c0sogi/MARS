import sys
import os
import numpy as np
import pandas as pd
import warnings

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.pipeline import RankingPipeline
from library.data_loader import load_data
from library.feature_engineering import MultiViewExtractor
from library.model_zoo import Stage1Ridge, Stage2LGBM
from library.utils import kendall_tau_metric


def main():
    # ---------------------------------------------------------
    # 1. Configuration for Fast Baseline
    # ---------------------------------------------------------
    # Enable Debug mode to limit training samples (2000 notebooks)
    # This ensures the script runs within the "Fast Baseline" time constraints.
    Config.DEBUG = True

    # Reduce boosting rounds for speed
    Config.NUM_BOOST_ROUND = 500

    # Silent mode for LGBM
    Config.VERBOSE_EVAL = -1

    # Ensure reproducibility
    np.random.seed(Config.SEED)

    # ---------------------------------------------------------
    # 2. Training Pipeline
    # ---------------------------------------------------------
    print("Initializing Pipeline...")
    pipeline = RankingPipeline()

    print("Running Training (Debug Mode)...")
    # load_cached_data=True allows using pre-computed features if available in ./working
    pipeline.run_train(load_cached_data=True)

    # ---------------------------------------------------------
    # 3. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n--- Validation Assessment ---")

    # Load validation data (Debug mode is True, so this loads the validation split of the debug subset)
    df_val = load_data(split="val", load_cached_data=True)

    # Generate features for validation
    extractor = MultiViewExtractor()
    feats_val = extractor.generate_features(df_val, split="val", load_cached_data=True)

    # Load trained models
    stage1 = Stage1Ridge()
    stage2 = Stage2LGBM()

    # Generate predictions
    # Stage 1: Ridge
    ridge_pred_val = stage1.predict(df_val)
    # Stage 2: LightGBM (Stacking Ridge preds + Multi-View Features)
    lgbm_pred_val = stage2.predict(df_val, ridge_pred_val, feats_val)

    # Post-process to get final cell orders
    val_submission = pipeline._post_process_sorting(df_val, lgbm_pred_val)

    # Construct Ground Truth for Validation
    # Sort by id and rank to reconstruct the correct order string
    df_val_sorted = df_val.sort_values(["id", "rank"])
    gt_series = df_val_sorted.groupby("id", observed=True)["cell_id"].apply(
        lambda x: " ".join(x)
    )
    df_val_gt = gt_series.reset_index()
    df_val_gt.columns = ["id", "cell_order"]

    # Calculate Metric
    score = kendall_tau_metric(df_val_gt, val_submission)

    # PRINT REQUIRED METRIC FORMAT
    print(f"Final Validation Metric: {score}")

    # ---------------------------------------------------------
    # 4. Failure Analysis
    # ---------------------------------------------------------
    print("\n--- Failure Analysis ---")

    # Filter for markdown cells in validation set as those are the ones we predict
    df_val_md = df_val[df_val["cell_type"] == "markdown"].copy()

    # Merge predictions
    # lgbm_pred_val: ['id', 'cell_id', 'pred_rank']
    analysis_df = df_val_md.merge(lgbm_pred_val, on=["id", "cell_id"], how="left")

    # Calculate Absolute Error
    analysis_df["error"] = (analysis_df["pct_rank"] - analysis_df["pred_rank"]).abs()

    # Merge features for correlation analysis
    # Ensure ID types match for merge (category vs string issues)
    feats_val_temp = feats_val.copy()
    feats_val_temp["id"] = feats_val_temp["id"].astype(str)
    analysis_df["id"] = analysis_df["id"].astype(str)

    analysis_df = analysis_df.merge(feats_val_temp, on=["id", "cell_id"], how="left")

    # Select numeric columns for correlation
    numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns

    # Compute correlation with Error
    if "error" in numeric_cols:
        correlations = (
            analysis_df[numeric_cols]
            .corrwith(analysis_df["error"])
            .sort_values(ascending=False)
        )
        print("Top features correlated with Error:")
        # Filter out the error column itself and print top 5 positive correlations
        print(correlations.drop("error", errors="ignore").head(5))
    else:
        print("Could not calculate correlations.")

    # ---------------------------------------------------------
    # 5. Submission Logic
    # ---------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if score > THRESHOLD:
        print(
            f"\nMetric ({score}) exceeds threshold ({THRESHOLD}). Generating submission for full test set..."
        )

        # Disable Debug mode to ensure we process the FULL test set, not a sample
        Config.DEBUG = False

        # Run Inference
        # load_cached_data=False forces the pipeline to read the raw test files
        # instead of loading a potentially partial cache from a previous debug run.
        pipeline.run_inference(load_cached_data=False)
    else:
        print(
            f"\nMetric ({score}) did not exceed threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()

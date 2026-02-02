import os
import sys
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_from_parquet
from library.train_tabular import run_tabular_training


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Optimize configuration for a fast baseline execution
    # We modify the class attributes directly to override the default settings.
    print("Configuring parameters for optimized execution...")

    # Increase LightGBM estimators for better convergence (Default: 10000)
    # We rely on early stopping, but set a high cap.
    Config.LGBM_PARAMS["n_estimators"] = 5000

    # -------------------------------------------------------------------------
    # 2. Run Base Model Training
    # -------------------------------------------------------------------------
    # Branch A: Tabular (Now the only branch)
    print("\n" + "=" * 40)
    print("Starting Tabular Training (Gradient Features Enhanced)")
    print("=" * 40)
    df_oof_tab, df_test_tab = run_tabular_training(debug=False)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Validation and Failure Analysis")
    print("=" * 40)

    # Load Validation Metadata to identify the specific hold-out validation set
    val_meta = pd.read_csv(Config.VAL_METADATA)
    val_ids = val_meta["segment_id"].values

    # Prepare Data
    # Rename columns for consistency with previous format
    df_results = df_oof_tab.rename(columns={"pred_time_to_eruption": "pred_final"})

    # Enforce non-negative constraint
    df_results["pred_final"] = np.maximum(df_results["pred_final"], 0)

    # Extract the subset corresponding to the hold-out validation set
    val_subset = df_results[df_results["segment_id"].isin(val_ids)].copy()

    if len(val_subset) == 0:
        print("Error: No validation segments matched in OOF predictions.")
        return

    # Calculate Final Validation Metric
    y_true_val = val_subset["true_time_to_eruption"]
    y_pred_val = val_subset["pred_final"]

    final_val_metric = mean_absolute_error(y_true_val, y_pred_val)
    print(f"Final Validation Metric: {final_val_metric}")

    # --- Failure Analysis ---
    print("\nPerforming Failure Analysis on Validation Set...")
    val_subset["error"] = (
        val_subset["true_time_to_eruption"] - val_subset["pred_final"]
    ).abs()

    # Load tabular features to correlate with error
    # We use the cache path helper to locate the feature file generated during tabular training
    val_feat_path = Config.get_cache_path("val", "parquet")

    if os.path.exists(val_feat_path):
        df_features = load_from_parquet(val_feat_path)

        # Merge features with error data
        analysis_df = pd.merge(
            val_subset[["segment_id", "error"]],
            df_features,
            on="segment_id",
            how="inner",
        )

        # Calculate correlations
        corrs = {}
        # Exclude ID and Error columns from feature list
        feature_cols = [
            c for c in analysis_df.columns if c not in ["segment_id", "error"]
        ]

        for col in feature_cols:
            # Ensure column is numeric before correlation
            if pd.api.types.is_numeric_dtype(analysis_df[col]):
                corrs[col] = analysis_df[col].corr(analysis_df["error"])

        # Identify top correlations
        corr_series = pd.Series(corrs)
        top_corrs = corr_series.abs().sort_values(ascending=False).head(5)

        print("Top 5 Feature Correlations with Error Magnitude:")
        print(top_corrs)
    else:
        print(
            f"Warning: Feature file not found at {val_feat_path}. Skipping correlation analysis."
        )

    # -------------------------------------------------------------------------
    # 4. Submission Generation
    # -------------------------------------------------------------------------
    THRESHOLD = 2587480.66

    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_metric}) meets the threshold ({THRESHOLD})."
        )
        print("Generating submission file...")

        # Save submission directly from Tabular predictions
        submission = df_test_tab.copy()
        # Ensure non-negative
        submission["time_to_eruption"] = np.maximum(submission["time_to_eruption"], 0)
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation metric ({final_val_metric}) does NOT meet the threshold ({THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()

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
from library.train_vision import run_vision_training
from library.stacking import train_meta_learner


def main():
    # -------------------------------------------------------------------------
    # 1. Setup and Configuration
    # -------------------------------------------------------------------------
    # Ensure reproducibility
    seed_everything(Config.SEED)

    # Optimize configuration for a fast baseline execution (< 2 hours)
    # We modify the class attributes directly to override the default settings.
    print("Configuring parameters for fast baseline execution...")

    # Reduce LightGBM estimators (Default: 10000)
    Config.LGBM_PARAMS["n_estimators"] = 1000

    # Reduce CNN epochs (Default: 20)
    Config.CNN_PARAMS["epochs"] = 5

    # Ensure batch size is appropriate for A100 (Default: 32 is safe, can go higher but 32 is fine)
    Config.CNN_PARAMS["batch_size"] = 32

    # -------------------------------------------------------------------------
    # 2. Run Base Model Training
    # -------------------------------------------------------------------------
    # Branch A: Tabular
    print("\n" + "=" * 40)
    print("Starting Branch A: Tabular Training")
    print("=" * 40)
    df_oof_tab, df_test_tab = run_tabular_training(debug=False)

    # Branch B: Vision
    print("\n" + "=" * 40)
    print("Starting Branch B: Vision Training")
    print("=" * 40)
    df_oof_vis, df_test_vis = run_vision_training(debug=False)

    # -------------------------------------------------------------------------
    # 3. Validation Assessment & Failure Analysis
    # -------------------------------------------------------------------------
    print("\n" + "=" * 40)
    print("Validation and Failure Analysis")
    print("=" * 40)

    # Load Validation Metadata to identify the specific hold-out validation set
    val_meta = pd.read_csv(Config.VAL_METADATA)
    val_ids = val_meta["segment_id"].values

    # Prepare Data for Meta-Learner Simulation
    # Rename columns for merging
    oof_tab = df_oof_tab.rename(columns={"pred_time_to_eruption": "pred_tab"})
    oof_vis = df_oof_vis.rename(columns={"pred_time_to_eruption": "pred_vis"})

    # Merge OOF predictions from both branches
    # Note: df_oof_* contains predictions for the entire dataset (Train + Val) via Cross-Validation
    df_meta_all = pd.merge(
        oof_tab[["segment_id", "pred_tab", "true_time_to_eruption"]],
        oof_vis[["segment_id", "pred_vis"]],
        on="segment_id",
        how="inner",
    )

    # Define Meta-Learner Features and Target
    X_meta = df_meta_all[["pred_tab", "pred_vis"]]
    y_meta = df_meta_all["true_time_to_eruption"]

    # Train a local Ridge Meta-Learner to replicate the stacking logic
    # This allows us to generate predictions and calculate metrics specifically for the validation set
    meta_model = Ridge(alpha=Config.META_RIDGE_ALPHA, random_state=Config.SEED)
    meta_model.fit(X_meta, y_meta)

    # Generate predictions on the OOF set
    meta_preds = meta_model.predict(X_meta)

    # Enforce non-negative constraint
    meta_preds = np.maximum(meta_preds, 0)

    df_meta_all["pred_final"] = meta_preds

    # Extract the subset corresponding to the hold-out validation set
    val_subset = df_meta_all[df_meta_all["segment_id"].isin(val_ids)].copy()

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
    THRESHOLD = 3135965.05

    if final_val_metric < THRESHOLD:
        print(
            f"\nValidation metric ({final_val_metric}) meets the threshold ({THRESHOLD})."
        )
        print("Generating submission file...")
        train_meta_learner(df_oof_tab, df_test_tab, df_oof_vis, df_test_vis)
    else:
        print(
            f"\nValidation metric ({final_val_metric}) does NOT meet the threshold ({THRESHOLD})."
        )
        print("Submission generation skipped.")


if __name__ == "__main__":
    main()

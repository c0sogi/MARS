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
    Config.LGBM_PARAMS["n_estimators"] = 5000

    # Reduce CNN epochs (Default: 20)
    Config.CNN_PARAMS["epochs"] = 15

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
    val_ids = set(val_meta["segment_id"].values)

    # Prepare Data for Meta-Learner Simulation
    # Rename columns for merging
    oof_tab = df_oof_tab.rename(columns={"pred_time_to_eruption": "pred_tab"})
    oof_vis = df_oof_vis.rename(columns={"pred_time_to_eruption": "pred_vis"})

    # Merge OOF predictions from both branches
    df_meta_all = pd.merge(
        oof_tab[["segment_id", "pred_tab", "true_time_to_eruption"]],
        oof_vis[["segment_id", "pred_vis"]],
        on="segment_id",
        how="inner",
    )

    # Split the OOF data into "Meta-Train" (Train set) and "Meta-Val" (Validation set)
    is_val = df_meta_all["segment_id"].isin(val_ids)
    train_subset = df_meta_all[~is_val].copy()
    val_subset = df_meta_all[is_val].copy()

    print(f"Meta-Learner Split: Train={len(train_subset)}, Val={len(val_subset)}")

    if len(val_subset) == 0:
        print("Error: No validation segments matched in OOF predictions.")
        return

    # Define Features and Target
    features = ["pred_tab", "pred_vis"]
    target = "true_time_to_eruption"

    # Train a local Ridge Meta-Learner ONLY on the training subset
    meta_model = Ridge(alpha=Config.META_RIDGE_ALPHA, random_state=Config.SEED)
    meta_model.fit(train_subset[features], train_subset[target])

    print(
        f"Meta-Learner Coefficients: Tabular={meta_model.coef_[0]:.4f}, Vision={meta_model.coef_[1]:.4f}"
    )

    # Predict ONLY on the validation subset
    val_preds = meta_model.predict(val_subset[features])

    # Enforce non-negative constraint
    val_preds = np.maximum(val_preds, 0)

    # Store predictions for analysis
    val_subset["pred_final"] = val_preds

    # Calculate Final Validation Metric
    final_val_metric = mean_absolute_error(val_subset[target], val_preds)
    print(f"Final Validation Metric (Unbiased): {final_val_metric}")

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

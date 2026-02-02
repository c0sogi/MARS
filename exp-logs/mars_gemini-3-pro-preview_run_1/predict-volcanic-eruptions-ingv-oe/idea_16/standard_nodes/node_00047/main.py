import os
import sys
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything
from library.engine_tabular import TabularTrainer
from library.engine_vision import VisionTrainer
from library.meta_learner import StackingEnsemble


def main():
    # 1. Configuration Adjustments for Fast Baseline
    # Adjust parameters to ensure execution within ~54 minutes
    # The dataset is small (~4000 samples), so we can use all data but reduce iterations.
    print("Configuring parameters for fast baseline execution...")
    Config.EPOCHS = 6  # Reduced from 35 to 6 for Vision
    Config.LGBM_PARAMS["n_estimators"] = 500  # Reduced from 2000 to 500 for Tabular

    # Ensure reproducibility
    seed_everything(Config.SEED)

    # 2. Run Tabular Branch
    print("\n" + "=" * 40)
    print("Starting Tabular Branch Pipeline")
    print("=" * 40)
    tabular_trainer = TabularTrainer()
    # load_cached_data=True allows skipping feature engineering if already done
    tabular_trainer.run_cv(load_cached_data=True)

    # 3. Run Vision Branch
    print("\n" + "=" * 40)
    print("Starting Vision Branch Pipeline")
    print("=" * 40)
    vision_trainer = VisionTrainer()
    # Vision trainer handles spectrogram caching internally
    vision_trainer.run_cv()

    # 4. Meta-Learner & Validation
    print("\n" + "=" * 40)
    print("Starting Meta-Learner & Validation")
    print("=" * 40)
    ensemble = StackingEnsemble()

    # Train meta-learner on OOF predictions and get the final metric
    final_mae = ensemble.train_meta_model()

    # REQUIRED OUTPUT: Print Final Validation Metric
    print(f"Final Validation Metric: {final_mae}")

    # 5. Failure Analysis
    print("\n" + "=" * 40)
    print("Performing Failure Analysis")
    print("=" * 40)

    try:
        # Load OOF Predictions
        df_tab_oof = pd.read_csv(os.path.join(Config.WORKING_DIR, "tabular_oof.csv"))
        df_vis_oof = pd.read_csv(os.path.join(Config.WORKING_DIR, "vision_oof.csv"))

        # Load Ground Truth
        df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
        df_val = pd.read_csv(Config.VAL_METADATA_PATH)
        df_gt = pd.concat([df_train, df_val], ignore_index=True)

        # Merge Predictions and GT
        df_merge = df_gt.merge(
            df_tab_oof, on=Config.SEGMENT_ID_COL, suffixes=("", "_tab")
        )
        df_merge = df_merge.merge(
            df_vis_oof, on=Config.SEGMENT_ID_COL, suffixes=("", "_vis")
        )

        # Rename columns for clarity (tabular_oof might have same col name as target)
        # The OOF files have columns: segment_id, time_to_eruption
        # We rename them to avoid collision with ground truth 'time_to_eruption'
        df_merge.rename(
            columns={
                "time_to_eruption_x": "target",  # From GT (if collision happened)
                "time_to_eruption_y": "pred_tabular",
                "time_to_eruption": "pred_vision",  # Depending on merge order/suffixes
            },
            inplace=True,
        )

        # Standardize column names if merge suffixes behaved differently
        if "time_to_eruption" in df_merge.columns:
            # This usually happens if suffixes weren't needed for one join
            # But let's be robust. We know GT is in df_gt.
            # Let's rebuild X and y cleanly using the ensemble logic
            pass

        # Re-construct X for ensemble prediction to get exact ensemble errors
        # We use the same alignment logic as MetaLearner
        df_tab_oof = df_tab_oof.rename(columns={Config.TARGET_COL: "pred_tabular"})
        df_vis_oof = df_vis_oof.rename(columns={Config.TARGET_COL: "pred_vision"})

        analysis_df = (
            df_gt[[Config.SEGMENT_ID_COL, Config.TARGET_COL]]
            .merge(df_tab_oof, on=Config.SEGMENT_ID_COL, how="inner")
            .merge(df_vis_oof, on=Config.SEGMENT_ID_COL, how="inner")
        )

        X_meta = analysis_df[["pred_tabular", "pred_vision"]].values
        y_true = analysis_df[Config.TARGET_COL].values

        # Predict using the trained ensemble model
        y_pred_ensemble = ensemble.model.predict(X_meta)

        # Calculate Error
        analysis_df["ensemble_pred"] = y_pred_ensemble
        analysis_df["abs_error"] = np.abs(
            analysis_df[Config.TARGET_COL] - analysis_df["ensemble_pred"]
        )

        # Load Features for Correlation Analysis
        # We load train and val features
        df_feat_train = pd.read_parquet(Config.TRAIN_FEATURES_PATH)
        df_feat_val = pd.read_parquet(Config.VAL_FEATURES_PATH)
        df_feats = pd.concat([df_feat_train, df_feat_val], ignore_index=True)

        # Merge features with error
        analysis_full = analysis_df.merge(
            df_feats, on=Config.SEGMENT_ID_COL, how="inner"
        )

        # Compute Correlations
        correlations = {}
        feature_cols = [
            c
            for c in df_feats.columns
            if c not in [Config.SEGMENT_ID_COL, Config.TARGET_COL]
        ]

        # Filter out non-numeric columns just in case
        valid_cols = [
            c for c in feature_cols if pd.api.types.is_numeric_dtype(analysis_full[c])
        ]

        for col in valid_cols:
            if analysis_full[col].std() > 1e-6:  # Avoid constant columns
                corr, _ = pearsonr(analysis_full["abs_error"], analysis_full[col])
                correlations[col] = corr
            else:
                correlations[col] = 0.0

        # Sort and Print Top Correlations
        sorted_corr = sorted(
            correlations.items(), key=lambda x: abs(x[1]), reverse=True
        )

        print("\nTop 10 Features Correlated with Error Magnitude:")
        for name, val in sorted_corr[:10]:
            print(f"{name}: {val:.4f}")

    except Exception as e:
        print(f"Failure analysis encountered an error: {e}")
        import traceback

        traceback.print_exc()

    # 6. Submission
    # Threshold: 1920624.12
    THRESHOLD = 1920624.12

    if final_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({final_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )
        ensemble.predict()
    else:
        print(
            f"\nValidation Metric ({final_mae}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()

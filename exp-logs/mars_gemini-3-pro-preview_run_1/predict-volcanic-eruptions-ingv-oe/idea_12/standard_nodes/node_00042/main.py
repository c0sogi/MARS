import os
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, mae_score, load_parquet
from library.model_tabular import run_lgbm_cv
from library.model_vision import train_cnn_fold, inference_cnn
from library.meta_learner import run_stacking, RidgeStacker
from library.feature_engineering import TabularFeatureEngineer


def main():
    # ---------------------------------------------------------
    # 1. Setup & Configuration
    # ---------------------------------------------------------
    seed_everything(Config.SEED)
    warnings.filterwarnings("ignore")

    # Fast Baseline Overrides to ensure execution within time limits
    # Reducing epochs and estimators while maintaining model structure
    Config.CNN_EPOCHS = 8
    Config.LGBM_PARAMS["n_estimators"] = 2000

    print("Initializing Dual-Resolution Spectral Energy Stacking Pipeline...")

    # ---------------------------------------------------------
    # 2. Load Metadata
    # ---------------------------------------------------------
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    test_meta = pd.read_csv(Config.TEST_METADATA_PATH)

    # Combine train and val for Full K-Fold CV
    # This ensures both Tabular and Vision branches use the same data split strategy
    full_meta = pd.concat([train_meta, val_meta], axis=0).reset_index(drop=True)

    # ---------------------------------------------------------
    # 3. Branch A: Tabular (LightGBM)
    # ---------------------------------------------------------
    print("\n=== Branch A: Tabular Model ===")
    # run_lgbm_cv handles feature engineering, caching, and CV
    # It returns OOF for the full_meta (train+val) and Test predictions
    tab_oof, tab_test = run_lgbm_cv(
        train_meta, val_meta, test_meta, load_cached_data=True
    )

    # ---------------------------------------------------------
    # 4. Branch B: Vision (CNN)
    # ---------------------------------------------------------
    print("\n=== Branch B: Vision Model ===")

    # Replicate the K-Fold split to generate aligned OOFs for Vision
    kf = KFold(n_splits=Config.NUM_FOLDS, shuffle=True, random_state=Config.SEED)

    vision_oof_list = []
    model_paths = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_meta)):
        print(f"--- Vision Fold {fold} ---")
        fold_train = full_meta.iloc[train_idx].reset_index(drop=True)
        fold_val = full_meta.iloc[val_idx].reset_index(drop=True)

        # Train CNN for this fold
        # train_cnn_fold returns (best_mae, oof_df) where oof_df contains predictions for fold_val
        _, fold_oof = train_cnn_fold(
            train_df=fold_train,
            val_df=fold_val,
            fold_idx=fold,
            load_cached_data=True,
            epochs=Config.CNN_EPOCHS,
        )

        vision_oof_list.append(fold_oof)
        model_paths.append(os.path.join(Config.CACHE_DIR, f"cnn_fold_{fold}.pth"))

    # Combine Vision OOFs into a single DataFrame
    vis_oof = pd.concat(vision_oof_list, axis=0).reset_index(drop=True)

    # Run Inference on Test Set using the ensemble of fold models
    vis_test = inference_cnn(test_meta, model_paths, load_cached_data=True)

    # ---------------------------------------------------------
    # 5. Validation & Metric Calculation
    # ---------------------------------------------------------
    print("\n=== Validation & Metric Calculation ===")

    # Align Data for Meta-Learner Training
    # Rename columns to avoid collisions
    tab_oof_clean = tab_oof.rename(columns={"time_to_eruption": "pred_tabular"})[
        ["segment_id", "pred_tabular"]
    ]

    vis_col = (
        "time_to_eruption_pred"
        if "time_to_eruption_pred" in vis_oof.columns
        else "time_to_eruption"
    )
    vis_oof_clean = vis_oof.rename(columns={vis_col: "pred_vision"})[
        ["segment_id", "pred_vision"]
    ]

    # Merge OOF predictions with Ground Truth
    stack_train = pd.merge(tab_oof_clean, vis_oof_clean, on="segment_id", how="inner")
    stack_train = pd.merge(
        stack_train,
        full_meta[["segment_id", "time_to_eruption"]],
        on="segment_id",
        how="inner",
    )

    # Train Ridge Meta-Learner locally to evaluate performance
    X_stack = stack_train[["pred_tabular", "pred_vision"]]
    y_stack = stack_train["time_to_eruption"]

    stacker = RidgeStacker()
    stacker.fit(X_stack, y_stack)

    # Generate Final OOF Predictions
    oof_final_preds = stacker.predict(X_stack)
    stack_train["pred_final"] = oof_final_preds

    # Filter for the Hold-out Validation Set
    # We evaluate specifically on the samples defined in 'val.csv'
    val_ids = val_meta["segment_id"].values
    val_results = stack_train[stack_train["segment_id"].isin(val_ids)].copy()

    # Calculate Final Metric
    final_metric = mae_score(val_results["time_to_eruption"], val_results["pred_final"])

    # Print exactly as requested
    print(f"Final Validation Metric: {final_metric}")

    # ---------------------------------------------------------
    # 6. Failure Analysis
    # ---------------------------------------------------------
    print("\n=== Failure Analysis ===")
    # Calculate Error Magnitude
    val_results["error"] = np.abs(
        val_results["time_to_eruption"] - val_results["pred_final"]
    )

    # Load Tabular Features for correlation analysis
    # run_lgbm_cv ensures 'val_features.parquet' exists in cache
    val_feat_path = os.path.join(Config.CACHE_DIR, "val_features.parquet")

    if os.path.exists(val_feat_path):
        val_features = load_parquet(val_feat_path)

        # Merge features with error data
        analysis_df = pd.merge(
            val_results[["segment_id", "error"]],
            val_features,
            on="segment_id",
            how="inner",
        )

        # Calculate Correlations
        numeric_cols = analysis_df.select_dtypes(include=[np.number]).columns
        correlations = (
            analysis_df[numeric_cols]
            .corrwith(analysis_df["error"])
            .sort_values(ascending=False)
        )

        print("Top 5 Features Correlated with Error:")
        # Head(6) because the top correlation is 'error' with itself (1.0)
        print(correlations.head(6))
    else:
        print("Warning: Validation features not found for failure analysis.")

    # ---------------------------------------------------------
    # 7. Submission
    # ---------------------------------------------------------
    THRESHOLD = 1920624.12

    if final_metric < THRESHOLD:
        print(
            f"\nMetric {final_metric} passed threshold {THRESHOLD}. Generating submission..."
        )
        # Execute run_stacking to generate and save the submission file
        run_stacking(tab_oof, tab_test, vis_oof, vis_test, full_meta)
    else:
        print(
            f"\nMetric {final_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()

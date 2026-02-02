import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Import necessary components from the provided library files
from library.config import Config
from library.utils import seed_everything, get_score, load_parquet
from library.feature_engineering import FeatureProcessor
from library.model_tabular import run_lgbm_cv
from library.model_vision import run_vision_cv
from library.model_stacking import MetaLearner


def main():
    # 1. Initialization and Configuration
    config = Config()
    seed_everything(config.SEED)

    print("Initializing Pipeline...")

    # 2. Feature Engineering
    # Process Train, Validation, and Test datasets
    # Using load_cached_data=True to leverage pre-computed features if available
    processor = FeatureProcessor()

    print("Processing Training Data...")
    processor.process_data(config.TRAIN_METADATA_PATH, "train", load_cached_data=False)

    print("Processing Validation Data...")
    processor.process_data(config.VAL_METADATA_PATH, "val", load_cached_data=False)

    print("Processing Test Data...")
    processor.process_data(config.TEST_METADATA_PATH, "test", load_cached_data=False)

    # 3. Train Base Models
    # Branch A: Tabular Model (LightGBM)
    # Uses the extensive feature set including Latent Source (PCA) and MFCCs
    print("Training Tabular Branch (LightGBM)...")
    df_oof_tab, df_test_tab = run_lgbm_cv(debug=False)

    # Branch B: Vision Model (EfficientNet)
    # Uses stacked Log-Mel Spectrograms
    print("Training Vision Branch (EfficientNet)...")
    df_oof_vis, df_test_vis = run_vision_cv(debug=False)

    # 4. Stacking (Meta-Learner)
    print("Training Meta-Learner (Ridge Stacking)...")

    # Prepare Meta-Learner Training Data
    # Rename columns for clarity
    df_oof_tab = df_oof_tab.rename(columns={"time_to_eruption": "pred_tabular"})
    df_oof_vis = df_oof_vis.rename(columns={"time_to_eruption": "pred_vision"})

    # Merge predictions from both branches
    df_train_meta = pd.merge(df_oof_tab, df_oof_vis, on="segment_id", how="inner")

    # Retrieve Ground Truth Targets
    # We combine train and val metadata to match the Cross-Validation OOF set
    df_meta_train = pd.read_csv(config.TRAIN_METADATA_PATH)
    df_meta_val = pd.read_csv(config.VAL_METADATA_PATH)
    df_meta_all = pd.concat([df_meta_train, df_meta_val], ignore_index=True)

    # Merge targets into the meta-training dataframe
    df_train_meta = pd.merge(
        df_train_meta,
        df_meta_all[["segment_id", "time_to_eruption"]],
        on="segment_id",
        how="inner",
    )
    df_train_meta = df_train_meta.rename(columns={"time_to_eruption": "target"})

    X_meta_train = df_train_meta[["pred_tabular", "pred_vision"]]
    y_meta_train = df_train_meta["target"]

    # Train the Ridge Meta-Learner using Nested CV to avoid leakage (Cite debug_lesson_3)
    kf = KFold(n_splits=5, shuffle=True, random_state=config.SEED)
    preds_oof_meta = np.zeros(len(X_meta_train))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_meta_train, y_meta_train)):
        X_train_fold = X_meta_train.iloc[train_idx]
        y_train_fold = y_meta_train.iloc[train_idx]
        X_val_fold = X_meta_train.iloc[val_idx]

        meta_learner_fold = MetaLearner()
        meta_learner_fold.train(X_train_fold, y_train_fold)
        preds_oof_meta[val_idx] = meta_learner_fold.predict(X_val_fold)

    # Train final meta-learner on full data for submission (ready for use if threshold is met)
    meta_learner = MetaLearner()
    meta_learner.train(X_meta_train, y_meta_train)

    # 5. Validation and Metrics
    final_mae = get_score(y_meta_train, preds_oof_meta)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_mae}")

    # 6. Failure Analysis
    print("\nPerforming Failure Analysis...")
    try:
        # Calculate absolute errors
        df_train_meta["error"] = np.abs(df_train_meta["target"] - preds_oof_meta)

        # Load features to correlate with error
        # We combine train and val features to match the meta-dataset
        train_feat_path = os.path.join(config.WORKING_DIR, "train_features.parquet")
        val_feat_path = os.path.join(config.WORKING_DIR, "val_features.parquet")

        if os.path.exists(train_feat_path) and os.path.exists(val_feat_path):
            df_feat_train = load_parquet(train_feat_path)
            df_feat_val = load_parquet(val_feat_path)
            df_feat_all = pd.concat([df_feat_train, df_feat_val], ignore_index=True)

            # Merge error with features
            df_analysis = pd.merge(
                df_train_meta[["segment_id", "error"]],
                df_feat_all,
                on="segment_id",
                how="inner",
            )

            # Calculate correlations with error
            # Exclude non-numeric columns (like segment_id which is categorical-ish)
            cols_to_corr = [
                c for c in df_analysis.columns if c not in ["segment_id", "error"]
            ]
            correlations = df_analysis[cols_to_corr].corrwith(df_analysis["error"])

            print("Top 5 Features correlated with Model Error:")
            print(correlations.abs().sort_values(ascending=False).head(5))
        else:
            print("Feature files not found for failure analysis.")

    except Exception as e:
        print(f"An error occurred during failure analysis: {e}")

    # 7. Conditional Submission
    THRESHOLD = 2250276.65

    if final_mae < THRESHOLD:
        print(
            f"\nValidation Metric ({final_mae}) meets threshold ({THRESHOLD}). Generating submission..."
        )

        # Prepare Test Data for Meta-Learner
        df_test_tab = df_test_tab.rename(columns={"time_to_eruption": "pred_tabular"})
        df_test_vis = df_test_vis.rename(columns={"time_to_eruption": "pred_vision"})

        # Merge test predictions
        df_test_meta = pd.merge(df_test_tab, df_test_vis, on="segment_id", how="inner")

        # Ensure alignment with Test Metadata (required for submission format)
        df_meta_test_file = pd.read_csv(config.TEST_METADATA_PATH)
        df_final_test = pd.merge(
            df_meta_test_file[["segment_id"]], df_test_meta, on="segment_id", how="left"
        )

        # Handle any potential missing predictions (though unlikely)
        df_final_test = df_final_test.fillna(0)

        X_test_meta = df_final_test[["pred_tabular", "pred_vision"]]

        # Generate Final Predictions
        final_preds = meta_learner.predict(X_test_meta)

        # Create Submission DataFrame
        submission_df = pd.DataFrame(
            {"segment_id": df_final_test["segment_id"], "time_to_eruption": final_preds}
        )

        # Save Submission
        os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")
        print(submission_df.head())

    else:
        print(
            f"\nValidation Metric ({final_mae}) does NOT meet threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()

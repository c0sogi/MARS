import os
import sys
import numpy as np
import pandas as pd
import joblib
import warnings

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, kendall_tau_metric, format_submission
from library.data_manager import get_partition_data
from library.vectorization import TextPipeline
from library.stage1_ridge import RidgeStacker
from library.feature_engineering import NeighborhoodFeatureExtractor
from library.stage2_lgbm import LGBMRanker

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # --------------------------------------------------------------------------
    # 1. Initialization & Setup
    # --------------------------------------------------------------------------
    seed_everything(Config.SEED)

    # Define a limit for training notebooks to ensure fast execution
    MAX_TRAIN_NOTEBOOKS = 10000

    print("Initializing pipeline components...")
    text_pipeline = TextPipeline()
    ridge_stacker = RidgeStacker()
    feature_extractor = NeighborhoodFeatureExtractor()
    lgbm_ranker = LGBMRanker()

    # --------------------------------------------------------------------------
    # 2. Data Loading (Train & Val)
    # --------------------------------------------------------------------------
    print("\n--- Data Loading ---")
    # Load Training Data
    df_train = get_partition_data("train", load_cached_data=True, debug=Config.DEBUG)

    # Subsample training data for fast baseline if not in debug mode (which already subsamples)
    if not Config.DEBUG:
        unique_train_ids = df_train["id"].unique()
        if len(unique_train_ids) > MAX_TRAIN_NOTEBOOKS:
            print(
                f"Subsampling training data to {MAX_TRAIN_NOTEBOOKS} notebooks for speed..."
            )
            # Use random choice with fixed seed
            rng = np.random.RandomState(Config.SEED)
            selected_ids = rng.choice(
                unique_train_ids, size=MAX_TRAIN_NOTEBOOKS, replace=False
            )
            df_train = df_train[df_train["id"].isin(selected_ids)].reset_index(
                drop=True
            )
            print(f"New training shape: {df_train.shape}")

    # Load Validation Data
    df_val = get_partition_data("val", load_cached_data=True, debug=Config.DEBUG)

    # --------------------------------------------------------------------------
    # 3. Text Pipeline (Vectorization)
    # --------------------------------------------------------------------------
    print("\n--- Text Vectorization ---")
    # Fit on training markdown sources
    train_md_sources = (
        df_train[df_train["cell_type"] == "markdown"]["source"].astype(str).tolist()
    )
    text_pipeline.fit(train_md_sources, load_cached_models=True)

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression
    # --------------------------------------------------------------------------
    print("\n--- Stage 1: Ridge Regression ---")

    # A. Generate OOF Predictions for Train
    # Check if we have cached OOF preds (optional optimization, but we'll run it to be safe/simple)
    # The RidgeStacker class doesn't cache OOFs internally to disk in the provided code,
    # so we run it.
    df_train_ridge_oof = ridge_stacker.train_and_predict_oof(df_train, text_pipeline)

    # B. Fit Final Ridge Model on Train
    ridge_stacker.fit(df_train, text_pipeline)

    # C. Predict on Validation
    df_val_ridge_preds = ridge_stacker.predict(df_val, text_pipeline)

    # --------------------------------------------------------------------------
    # 5. Stage 2: Feature Engineering
    # --------------------------------------------------------------------------
    print("\n--- Stage 2: Feature Engineering ---")

    # Extract features for Train
    df_train_feats = feature_extractor.extract_features(
        df_train,
        text_pipeline,
        df_train_ridge_oof,
        partition="train",
        load_cached_data=True,
    )

    # Extract features for Validation
    df_val_feats = feature_extractor.extract_features(
        df_val,
        text_pipeline,
        df_val_ridge_preds,
        partition="val",
        load_cached_data=True,
    )

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM Training
    # --------------------------------------------------------------------------
    print("\n--- Stage 2: LightGBM Training ---")
    lgbm_ranker.train_model(df_train_feats, df_val_feats)

    # --------------------------------------------------------------------------
    # 7. Validation Inference & Metric Calculation
    # --------------------------------------------------------------------------
    print("\n--- Validation Evaluation ---")

    # Predict ranks for validation set
    val_preds_rank = lgbm_ranker.predict_rank(df_val_feats)
    df_val_feats["pred_rank"] = val_preds_rank

    # Map predictions back to the validation dataframe cells
    # We need to reconstruct the order.
    # 1. Get code cells (anchors) with their fixed ranks.
    # 2. Get markdown cells with predicted ranks.
    # 3. Sort.

    # Create a dictionary of predictions: id -> list of cell_ids
    val_predictions = {}

    # We iterate over unique notebooks in validation set
    # We can use the original df_val to get code cells

    # Helper to speed up lookup: map notebook_id to dataframe slice
    # But groupby is efficient enough

    # First, merge predictions into df_val for markdown cells
    # df_val_feats contains ['cell_id', 'pred_rank']
    # We left join this into df_val
    df_val_merged = df_val.merge(
        df_val_feats[["cell_id", "pred_rank"]], on="cell_id", how="left"
    )

    grouped_val = df_val_merged.groupby("id")

    for nb_id, group in grouped_val:
        # Separate code and markdown
        code_cells = group[group["cell_type"] == "code"].copy()
        md_cells = group[group["cell_type"] == "markdown"].copy()

        # Assign ranks to code cells
        n_code = len(code_cells)
        if n_code == 0:
            # Edge case: no code cells. Just sort markdown by predicted rank.
            # Ranks are already in md_cells['pred_rank']
            pass
        elif n_code == 1:
            code_cells["final_rank"] = 0.0
        else:
            code_cells["final_rank"] = np.linspace(0.0, 1.0, n_code)

        # Markdown cells use predicted rank
        md_cells["final_rank"] = md_cells["pred_rank"]

        # Combine
        combined = pd.concat([code_cells, md_cells])

        # Sort by final_rank
        combined_sorted = combined.sort_values("final_rank")

        # Store order
        val_predictions[nb_id] = combined_sorted["cell_id"].tolist()

    # Load Ground Truth
    df_val_gt = pd.read_csv(Config.VAL_METADATA_PATH)

    # Compute Metric
    score = kendall_tau_metric(df_val_gt, val_predictions)
    print(f"Final Validation Metric: {score}")

    # --------------------------------------------------------------------------
    # 8. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Analyze correlation between error and features on Validation set
    # Error = abs(pred_rank - true_rank)
    # We have 'pct_rank' (true) and 'pred_rank' in df_val_feats (since we assigned it earlier)
    # Note: df_val_feats was constructed from df_val which has 'pct_rank'

    df_analysis = df_val_feats.copy()
    df_analysis["error"] = np.abs(df_analysis["pct_rank"] - df_analysis["pred_rank"])

    features_to_corr = [
        "n_code",
        "n_md",
        "code_ratio",
        "lex_std",
        "lat_std",
        "lex_max",
        "lat_max",
    ]

    print("Correlation of Error with Features:")
    correlations = (
        df_analysis[features_to_corr + ["error"]].corr()["error"].drop("error")
    )
    print(correlations.sort_values(ascending=False))

    # --------------------------------------------------------------------------
    # 9. Test Inference & Submission
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if score > THRESHOLD:
        print("\n--- Validation Score meets threshold. Generating Submission... ---")

        # Load Test Data
        df_test = get_partition_data("test", load_cached_data=True, debug=Config.DEBUG)

        # Stage 1 Predict
        df_test_ridge_preds = ridge_stacker.predict(df_test, text_pipeline)

        # Stage 2 Features
        df_test_feats = feature_extractor.extract_features(
            df_test,
            text_pipeline,
            df_test_ridge_preds,
            partition="test",
            load_cached_data=True,
        )

        # Stage 2 Predict
        test_preds_rank = lgbm_ranker.predict_rank(df_test_feats)
        df_test_feats["pred_rank"] = test_preds_rank

        # Merge preds back to test df
        df_test_merged = df_test.merge(
            df_test_feats[["cell_id", "pred_rank"]], on="cell_id", how="left"
        )

        # Sort and Format
        test_predictions = {}
        grouped_test = df_test_merged.groupby("id")

        for nb_id, group in grouped_test:
            code_cells = group[group["cell_type"] == "code"].copy()
            md_cells = group[group["cell_type"] == "markdown"].copy()

            # Code Ranks
            # In test, we rely on the order they appear in the JSON (which is correct relative order)
            # We must ensure we respect that order.
            # NotebookLoader.parse_notebook assigns 'code_rank' for this purpose.
            if "code_rank" in code_cells.columns:
                code_cells = code_cells.sort_values("code_rank")

            n_code = len(code_cells)
            if n_code == 0:
                pass
            elif n_code == 1:
                code_cells["final_rank"] = 0.0
            else:
                code_cells["final_rank"] = np.linspace(0.0, 1.0, n_code)

            md_cells["final_rank"] = md_cells["pred_rank"]

            combined = pd.concat([code_cells, md_cells])
            combined_sorted = combined.sort_values("final_rank")

            test_predictions[nb_id] = combined_sorted["cell_id"].tolist()

        # Create Submission File
        format_submission(test_predictions, save_path=Config.SUBMISSION_PATH)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation Score {score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

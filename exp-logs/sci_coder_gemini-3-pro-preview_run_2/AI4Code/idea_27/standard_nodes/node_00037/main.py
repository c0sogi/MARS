import os
import sys
import numpy as np
import pandas as pd
import warnings
import gc

# Import from provided libraries
from library.config import Config
from library.utils import set_seed, compute_kendall_tau
from library.data_manager import NotebookLoader
from library.vectorization import TextProcessor
from library.feature_extraction import FeatureEngineer
from library.models import Stage1Ridge, Stage2LGBM

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    # 1. Initialization
    config = Config()
    set_seed(42)
    print("Initializing pipeline...")

    # 2. Data Loading
    # We load train and val separately to perform strict hold-out validation
    loader = NotebookLoader(config)
    print("Loading datasets...")
    df_train_corpus, df_val_corpus = loader.prepare_datasets(load_cached_data=True)

    # 3. Vectorization
    # Fit on combined corpus for better vocabulary coverage, then transform separately
    print("Fitting vectorizers...")
    df_full_corpus = pd.concat([df_train_corpus, df_val_corpus], ignore_index=True)
    processor = TextProcessor(config)
    processor.fit_pipeline(df_full_corpus, load_cached_models=True)

    print("Transforming training data...")
    tfidf_train, svd_train = processor.transform_cells(
        df_train_corpus, mode="train", load_cached_data=True
    )

    print("Transforming validation data...")
    tfidf_val, svd_val = processor.transform_cells(
        df_val_corpus, mode="val", load_cached_data=True
    )

    # Clean up to save memory
    del df_full_corpus
    gc.collect()

    # 4. Feature Extraction
    engineer = FeatureEngineer(config)

    print("Extracting features for training set...")
    df_feat_train = engineer.extract_features(
        df_train_corpus, tfidf_train, svd_train, mode="train", load_cached_data=True
    )

    print("Extracting features for validation set...")
    df_feat_val = engineer.extract_features(
        df_val_corpus, tfidf_val, svd_val, mode="val", load_cached_data=True
    )

    # 5. Stage 1: Ridge Regression
    print("Training Stage 1 (Ridge)...")
    stage1 = Stage1Ridge(config)

    # Fit on Train, get OOF for Train
    train_oof = stage1.fit_oof(
        df_train_corpus, tfidf_train, df_feat_train, load_cached_data=True
    )
    df_feat_train["ridge_pred"] = train_oof

    # Predict on Val
    val_ridge_preds = stage1.predict(df_val_corpus, tfidf_val, df_feat_val)
    df_feat_val["ridge_pred"] = val_ridge_preds

    # 6. Stage 2: LightGBM
    print("Training Stage 2 (LightGBM)...")
    stage2 = Stage2LGBM(config)

    # Fit on Train
    stage2.fit(df_feat_train, load_cached_data=True)

    # Predict on Val
    val_final_preds = stage2.predict(df_feat_val)
    df_feat_val["pred_rank"] = val_final_preds

    # 7. Validation Assessment
    print("Performing validation assessment...")

    # Reconstruct cell orders
    # We need to map predictions back to notebooks
    # df_feat_val contains 'id', 'cell_id', 'pred_rank'

    # Create a dictionary for fast lookup of predicted ranks
    pred_map = dict(zip(df_feat_val["cell_id"], df_feat_val["pred_rank"]))

    # Prepare lists for metric computation
    val_ids = []
    val_orders = []

    # Group by notebook to sort
    # We iterate over the validation corpus to ensure we have all cells (code + markdown)
    for nb_id, group in df_val_corpus.groupby("id"):
        cells = group.copy()

        # Assign ranks
        # Code cells: Fixed anchors
        code_mask = cells["cell_type"] == "code"
        n_code = code_mask.sum()
        if n_code > 0:
            cells.loc[code_mask, "rank"] = np.linspace(0, 1, n_code)
        else:
            # Fallback if no code cells (rare)
            pass

        # Markdown cells: Predicted ranks
        md_mask = cells["cell_type"] == "markdown"
        # Map predictions. Fillna with 0.5 or similar if missing (shouldn't happen)
        cells.loc[md_mask, "rank"] = (
            cells.loc[md_mask, "cell_id"].map(pred_map).fillna(0.5)
        )

        # Sort
        cells = cells.sort_values("rank")
        cell_order = " ".join(cells["cell_id"].tolist())

        val_ids.append(nb_id)
        val_orders.append(cell_order)

    df_val_preds = pd.DataFrame({"id": val_ids, "cell_order": val_orders})

    # Load Ground Truth
    df_val_meta = pd.read_csv(os.path.join(config.METADATA_DIR, "val_metadata.csv"))
    df_val_gt = df_val_meta[["id", "cell_order"]]

    # Compute Metric
    kt_score = compute_kendall_tau(df_val_gt, df_val_preds)
    print(f"Final Validation Metric: {kt_score}")

    # 8. Failure Analysis
    print("\n--- Failure Analysis ---")
    # Calculate error for markdown cells
    df_feat_val["error"] = np.abs(df_feat_val["target"] - df_feat_val["pred_rank"])

    # Correlate error with features
    # Select numeric columns only
    numeric_cols = df_feat_val.select_dtypes(include=[np.number]).columns
    correlations = (
        df_feat_val[numeric_cols]
        .corrwith(df_feat_val["error"])
        .sort_values(ascending=False)
    )

    print("Correlation of features with Prediction Error (Top 5 Positive):")
    print(correlations.head(5))
    print("Correlation of features with Prediction Error (Top 5 Negative):")
    print(correlations.tail(5))

    # 9. Submission
    THRESHOLD = 0.7959051868218839

    if kt_score > THRESHOLD:
        print(f"\nValidation score {kt_score} > {THRESHOLD}. Generating submission...")

        # Load Test Data
        print("Loading test data...")
        df_test_corpus = loader.load_test_data(load_cached_data=True)

        # Transform
        print("Transforming test data...")
        tfidf_test, svd_test = processor.transform_cells(
            df_test_corpus, mode="test", load_cached_data=True
        )

        # Extract Features
        print("Extracting test features...")
        df_feat_test = engineer.extract_features(
            df_test_corpus, tfidf_test, svd_test, mode="test", load_cached_data=True
        )

        # Predict Stage 1
        print("Predicting Stage 1 (Test)...")
        test_ridge_preds = stage1.predict(df_test_corpus, tfidf_test, df_feat_test)
        df_feat_test["ridge_pred"] = test_ridge_preds

        # Predict Stage 2
        print("Predicting Stage 2 (Test)...")
        test_final_preds = stage2.predict(df_feat_test)
        df_feat_test["pred_rank"] = test_final_preds

        # Sort and Format
        print("Formatting submission...")
        submission_rows = []
        pred_map_test = dict(zip(df_feat_test["cell_id"], df_feat_test["pred_rank"]))

        for nb_id, group in df_test_corpus.groupby("id"):
            cells = group.copy()

            # Code cells
            code_mask = cells["cell_type"] == "code"
            n_code = code_mask.sum()
            if n_code > 0:
                cells.loc[code_mask, "rank"] = np.linspace(0, 1, n_code)

            # Markdown cells
            md_mask = cells["cell_type"] == "markdown"
            cells.loc[md_mask, "rank"] = (
                cells.loc[md_mask, "cell_id"].map(pred_map_test).fillna(0.5)
            )

            # Sort
            cells = cells.sort_values("rank")
            cell_order = " ".join(cells["cell_id"].tolist())

            submission_rows.append({"id": nb_id, "cell_order": cell_order})

        df_submission = pd.DataFrame(submission_rows)
        df_submission.to_csv(config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score {kt_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

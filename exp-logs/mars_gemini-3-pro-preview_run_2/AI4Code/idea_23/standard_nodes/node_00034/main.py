import os
import sys
import numpy as np
import pandas as pd
import random
import torch
import warnings
from scipy.stats import pearsonr

# Import provided library modules
from library.config import Config
from library.data_loader import NotebookLoader
from library.preprocessor import TextPipeline
from library.feature_engineering import FeatureEngine
from library.models import Stage1Ridge, Stage2LGBM
from library.metrics import compute_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    # --------------------------------------------------------------------------
    # 1. Setup & Configuration
    # --------------------------------------------------------------------------
    set_seed(Config.SEED)
    Config.setup()

    print("Starting End-to-End Pipeline...")

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    loader = NotebookLoader()

    # Load Train Data (Subsample to 50% for speed as per requirements)
    df_train = loader.load_train_data(load_cached_data=True, sample_fraction=0.5)

    # Load Validation Data (Full set for accurate metric)
    df_val = loader.load_val_data(load_cached_data=True)

    print(f"Train data shape: {df_train.shape}")
    print(f"Val data shape: {df_val.shape}")

    # --------------------------------------------------------------------------
    # 3. Text Vectorization (TF-IDF & SVD)
    # --------------------------------------------------------------------------
    text_pipeline = TextPipeline()

    # Fit on training data and transform both train and val
    # Note: fit_transform_corpus returns SVD features
    svd_train = text_pipeline.fit_transform_corpus(df_train, load_cached_models=True)
    svd_val = text_pipeline.transform(df_val["source"].astype(str).fillna(""))

    # We also need sparse TF-IDF features for Stage 1 Ridge
    # Access the fitted vectorizer from the pipeline
    print("Extracting sparse TF-IDF features for Stage 1...")
    tfidf_train = text_pipeline.vectorizer.transform(
        df_train["source"].astype(str).fillna("")
    )
    tfidf_val = text_pipeline.vectorizer.transform(
        df_val["source"].astype(str).fillna("")
    )

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression (Baseline)
    # --------------------------------------------------------------------------
    stage1_model = Stage1Ridge()

    # Prepare targets and groups for OOF
    y_train = df_train["rank"].values
    groups_train = df_train["ancestor_id"].values

    # Generate OOF predictions for Train (for Stage 2 features)
    # Filter for markdown cells only for training, but Ridge sees all text usually?
    # The task is to rank cells. Usually we train on all cells or just markdown?
    # The target 'rank' exists for all cells.
    # However, in Stage 2 we only predict for Markdown cells.
    # Let's generate OOF preds for ALL cells in df_train so FeatureEngine can use them if needed,
    # though FeatureEngine mainly uses Ridge preds for the markdown cells themselves.

    oof_preds_train = stage1_model.get_oof_predictions(
        tfidf_train, y_train, groups_train, load_cached_data=True
    )

    # Fit Ridge on full training data for inference
    stage1_model.fit(tfidf_train, y_train)

    # Predict on Validation
    preds_val_stage1 = stage1_model.predict(tfidf_val)

    # Create dictionaries for FeatureEngine
    # Map cell_id -> prediction
    train_ridge_preds_map = dict(zip(df_train["cell_id"].values, oof_preds_train))
    val_ridge_preds_map = dict(zip(df_val["cell_id"].values, preds_val_stage1))

    # --------------------------------------------------------------------------
    # 5. Feature Engineering (Stage 2)
    # --------------------------------------------------------------------------
    feature_engine = FeatureEngine()

    # Generate dense features for Markdown cells
    # Note: FeatureEngine handles the logic of using code cells as anchors

    # Train Features
    df_feats_train = feature_engine.create_stage2_features(
        df_train, svd_train, train_ridge_preds_map, mode="train", load_cached_data=True
    )

    # Val Features
    df_feats_val = feature_engine.create_stage2_features(
        df_val, svd_val, val_ridge_preds_map, mode="train", load_cached_data=True
    )

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM (Refinement)
    # --------------------------------------------------------------------------
    stage2_model = Stage2LGBM()

    # Prepare X and y for LightGBM
    # Drop non-feature columns
    drop_cols = ["cell_id", "target"]
    feature_cols = [c for c in df_feats_train.columns if c not in drop_cols]

    X_train_lgbm = df_feats_train[feature_cols]
    y_train_lgbm = df_feats_train["target"]

    X_val_lgbm = df_feats_val[feature_cols]
    y_val_lgbm = df_feats_val["target"]

    # Train
    stage2_model.fit(X_train_lgbm, y_train_lgbm, X_val_lgbm, y_val_lgbm)

    # Predict on Val
    preds_val_stage2 = stage2_model.predict(X_val_lgbm)

    # --------------------------------------------------------------------------
    # 7. Validation & Metric Calculation
    # --------------------------------------------------------------------------
    print("Calculating Validation Metrics...")

    # We need to reconstruct the order for each notebook in validation
    # 1. Get Code cells (fixed order/rank)
    # 2. Get Markdown cells (predicted rank)
    # 3. Sort

    # Create a map of predicted ranks for markdown cells in val
    val_md_preds_map = dict(zip(df_feats_val["cell_id"].values, preds_val_stage2))

    val_predictions = {}

    # Group df_val by notebook id
    for nb_id, group in df_val.groupby("id"):
        cells = []
        for _, row in group.iterrows():
            cid = row["cell_id"]
            ctype = row["cell_type"]

            if ctype == "code":
                # Code cells keep their ground truth rank relative to each other.
                # In the problem description, code cells are in correct order.
                # We can use their provided rank or just their relative order.
                # To mix with predicted markdown ranks, we use the rank from df_val
                # (which is normalized 0..1 based on ground truth).
                rank = row["rank"]
            else:
                # Markdown cell: use prediction
                rank = val_md_preds_map.get(cid, 0.5)  # Default to middle if missing

            cells.append((cid, rank))

        # Sort by rank
        cells.sort(key=lambda x: x[1])
        sorted_ids = [c[0] for c in cells]
        val_predictions[nb_id] = sorted_ids

    # Compute Kendall Tau
    final_score = compute_score(df_val, val_predictions)
    print(f"Final Validation Metric: {final_score}")

    # --------------------------------------------------------------------------
    # 8. Failure Analysis
    # --------------------------------------------------------------------------
    print("\n--- Failure Analysis ---")
    # Calculate error for markdown cells
    errors = np.abs(y_val_lgbm - preds_val_stage2)

    # Correlate errors with features
    analysis_df = X_val_lgbm.copy()
    analysis_df["error"] = errors

    correlations = {}
    for col in feature_cols:
        # Skip if column is constant
        if analysis_df[col].nunique() <= 1:
            continue
        corr, _ = pearsonr(analysis_df[col], analysis_df["error"])
        correlations[col] = corr

    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

    print("Top 5 Features correlated with Error Magnitude:")
    for name, val in sorted_corr[:5]:
        print(f"  {name}: {val:.4f}")

    # --------------------------------------------------------------------------
    # 9. Submission Generation
    # --------------------------------------------------------------------------
    THRESHOLD = 0.7959051868218839

    if final_score > THRESHOLD:
        print("\nValidation score meets threshold. Generating submission...")

        # Load Test Data
        df_test = loader.load_test_data(load_cached_data=True)

        # 1. Vectorize Test Text
        print("Vectorizing test data...")
        svd_test = text_pipeline.transform(df_test["source"].astype(str).fillna(""))
        tfidf_test = text_pipeline.vectorizer.transform(
            df_test["source"].astype(str).fillna("")
        )

        # 2. Stage 1 Prediction
        print("Running Stage 1 inference on test...")
        preds_test_stage1 = stage1_model.predict(tfidf_test)
        test_ridge_preds_map = dict(zip(df_test["cell_id"].values, preds_test_stage1))

        # 3. Feature Engineering
        print("Generating Stage 2 features for test...")
        df_feats_test = feature_engine.create_stage2_features(
            df_test, svd_test, test_ridge_preds_map, mode="test", load_cached_data=True
        )

        # 4. Stage 2 Prediction
        print("Running Stage 2 inference on test...")
        X_test_lgbm = df_feats_test[feature_cols]
        preds_test_stage2 = stage2_model.predict(X_test_lgbm)
        test_md_preds_map = dict(
            zip(df_feats_test["cell_id"].values, preds_test_stage2)
        )

        # 5. Construct Final Order
        print("Constructing final cell orders...")
        submission_rows = []

        # The test dataframe contains all cells. Code cells are in correct relative order.
        # We need to assign ranks to code cells to interleave them with markdown.

        for nb_id, group in df_test.groupby("id"):
            # Identify code cells to establish skeleton
            # In df_test, rows are ordered by appearance in JSON.
            # For code cells, this is the correct relative order.

            # Separate code and markdown
            code_cells = group[group["cell_type"] == "code"].copy()
            md_cells = group[group["cell_type"] == "markdown"].copy()

            # Assign ranks to code cells: Equidistant 0.0 to 1.0
            n_code = len(code_cells)
            if n_code > 0:
                # e.g., if 3 code cells: 0.0, 0.5, 1.0
                if n_code == 1:
                    code_ranks = [0.0]
                else:
                    code_ranks = np.linspace(0.0, 1.0, n_code)
            else:
                code_ranks = []

            final_cells = []

            # Add code cells with fixed ranks
            for i, (_, row) in enumerate(code_cells.iterrows()):
                final_cells.append((row["cell_id"], code_ranks[i]))

            # Add markdown cells with predicted ranks
            for _, row in md_cells.iterrows():
                pred_rank = test_md_preds_map.get(row["cell_id"], 0.5)
                final_cells.append((row["cell_id"], pred_rank))

            # Sort
            final_cells.sort(key=lambda x: x[1])
            ordered_ids = [c[0] for c in final_cells]

            submission_rows.append({"id": nb_id, "cell_order": " ".join(ordered_ids)})

        # Save Submission
        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        print(
            f"\nValidation score {final_score} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

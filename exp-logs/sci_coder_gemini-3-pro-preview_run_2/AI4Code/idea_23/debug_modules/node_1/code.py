import os
import sys
import numpy as np
import pandas as pd
import random
import warnings

# Import library modules
from library.config import Config
from library.data_loader import NotebookLoader
from library.preprocessor import TextPipeline
from library.models import Stage1Ridge, Stage2LGBM
from library.feature_engineering import FeatureEngine
from library.metrics import compute_score

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== Starting Demo Script ===")
    set_seed(42)

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    # We modify the Config class attributes directly before instantiation
    # to run a lightweight version of the pipeline.
    print("Configuring hyperparameters for fast demonstration...")
    Config.TFIDF_VOCAB_SIZE = 1000
    Config.SVD_N_COMPONENTS = 10
    Config.ANCHOR_CONTENT_DIMS = 5  # Must be <= SVD_N_COMPONENTS
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 5
    Config.WORKING_DIR = "./working/demo_run"  # Isolate demo output
    Config.setup()

    # --------------------------------------------------------------------------
    # 2. Data Loading
    # --------------------------------------------------------------------------
    loader = NotebookLoader()

    # Load a small sample of training data (5%)
    print("\n--- Loading Data ---")
    df_train = loader.load_train_data(load_cached_data=False, sample_fraction=0.05)

    # Load validation data and slice manually to keep it fast (e.g., 20 notebooks)
    df_val_full = loader.load_val_data(load_cached_data=False)
    val_ids = df_val_full["id"].unique()[:20]
    df_val = df_val_full[df_val_full["id"].isin(val_ids)].reset_index(drop=True)

    # Load test data and slice manually
    df_test_full = loader.load_test_data(load_cached_data=False)
    test_ids = df_test_full["id"].unique()[:20]
    df_test = df_test_full[df_test_full["id"].isin(test_ids)].reset_index(drop=True)

    print(f"Train shape: {df_train.shape}")
    print(f"Val shape:   {df_val.shape}")
    print(f"Test shape:  {df_test.shape}")

    assert not df_train.empty, "Training dataframe is empty."
    assert not df_val.empty, "Validation dataframe is empty."

    # --------------------------------------------------------------------------
    # 3. Preprocessing (Text Pipeline)
    # --------------------------------------------------------------------------
    print("\n--- Text Processing (TF-IDF + SVD) ---")
    pipeline = TextPipeline()

    # Fit on Train, Transform Train
    # Note: fit_transform_corpus returns SVD features
    svd_train = pipeline.fit_transform_corpus(df_train, load_cached_models=False)

    # We also need the sparse TF-IDF matrix for the Stage 1 Ridge model
    # Accessing the vectorizer directly after it has been fitted
    tfidf_train = pipeline.vectorizer.transform(
        df_train["source"].astype(str).fillna("")
    )

    # Transform Validation and Test
    svd_val = pipeline.transform(df_val["source"].astype(str).fillna(""))
    tfidf_val = pipeline.vectorizer.transform(df_val["source"].astype(str).fillna(""))

    svd_test = pipeline.transform(df_test["source"].astype(str).fillna(""))
    tfidf_test = pipeline.vectorizer.transform(df_test["source"].astype(str).fillna(""))

    assert svd_train.shape[1] == Config.SVD_N_COMPONENTS, "SVD dimension mismatch."
    assert tfidf_train.shape[0] == len(df_train), "TF-IDF row count mismatch."

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression
    # --------------------------------------------------------------------------
    print("\n--- Stage 1: Ridge Regression ---")
    ridge_model = Stage1Ridge()

    # A. Get Out-Of-Fold Predictions for Training (to prevent leakage in Stage 2)
    # Ranks and Groups needed
    y_train = df_train["rank"].values
    groups_train = df_train["ancestor_id"].values

    oof_preds = ridge_model.get_oof_predictions(
        tfidf_train, y_train, groups_train, load_cached_data=False
    )

    # B. Fit on full training data for inference
    ridge_model.fit(tfidf_train, y_train)

    # C. Predict on Validation and Test
    stage1_val_preds = ridge_model.predict(tfidf_val)
    stage1_test_preds = ridge_model.predict(tfidf_test)

    assert len(oof_preds) == len(df_train), "OOF preds length mismatch."

    # --------------------------------------------------------------------------
    # 5. Feature Engineering for Stage 2
    # --------------------------------------------------------------------------
    print("\n--- Stage 2: Feature Engineering ---")
    fe = FeatureEngine()

    # Prepare dictionaries mapping cell_id -> stage1_prediction
    # Train uses OOF predictions
    train_pred_dict = dict(zip(df_train["cell_id"], oof_preds))
    # Val/Test use model predictions
    val_pred_dict = dict(zip(df_val["cell_id"], stage1_val_preds))
    test_pred_dict = dict(zip(df_test["cell_id"], stage1_test_preds))

    # Generate Features
    # Note: FeatureEngine filters for Markdown cells internally
    df_feats_train = fe.create_stage2_features(
        df_train, svd_train, train_pred_dict, mode="train", load_cached_data=False
    )
    df_feats_val = fe.create_stage2_features(
        df_val, svd_val, val_pred_dict, mode="train", load_cached_data=False
    )
    df_feats_test = fe.create_stage2_features(
        df_test, svd_test, test_pred_dict, mode="test", load_cached_data=False
    )

    print(f"Stage 2 Train Features: {df_feats_train.shape}")

    # Verify expected columns exist
    expected_col = "neigh_mean_rank"
    assert expected_col in df_feats_train.columns, f"Missing feature {expected_col}"

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM
    # --------------------------------------------------------------------------
    print("\n--- Stage 2: LightGBM ---")
    lgbm_model = Stage2LGBM()

    # Prepare X and y
    # Drop non-feature columns
    drop_cols = ["cell_id", "target"]
    X_train_s2 = df_feats_train.drop(columns=drop_cols)
    y_train_s2 = df_feats_train["target"]

    X_val_s2 = df_feats_val.drop(columns=drop_cols)
    y_val_s2 = df_feats_val["target"]

    X_test_s2 = df_feats_test.drop(columns=drop_cols)

    # Fit Model
    lgbm_model.fit(X_train_s2, y_train_s2, X_val_s2, y_val_s2)

    # Predict
    final_val_preds = lgbm_model.predict(X_val_s2)
    final_test_preds = lgbm_model.predict(X_test_s2)

    assert len(final_val_preds) == len(
        df_feats_val
    ), "Validation prediction length mismatch."

    # --------------------------------------------------------------------------
    # 7. Post-Processing & Evaluation
    # --------------------------------------------------------------------------
    print("\n--- Evaluation ---")

    # Helper to reconstruct order from predictions
    def reconstruct_order(df_base, features_df, preds):
        # 1. Create a map of markdown cell_id -> predicted rank
        pred_map = dict(zip(features_df["cell_id"], preds))

        # 2. Get all cells for each notebook
        results = {}
        for nb_id, group in df_base.groupby("id"):
            cells = []
            for _, row in group.iterrows():
                cid = row["cell_id"]
                ctype = row["cell_type"]

                # If markdown, get predicted rank. If code, use its inherent rank (or approximate)
                # In this pipeline, code cells are anchors.
                # For evaluation purposes on Train/Val, we know the ground truth rank of code cells.
                # For Test, we assume code cells are already in order (0, 1, 2...).

                if ctype == "markdown":
                    rank = pred_map.get(cid, 0.5)  # Default to middle if missing
                else:
                    # In this demo, we use the 'rank' column which is populated for Train/Val
                    # For test, it is -1.0, so we use the index in the group as proxy
                    if row["rank"] != -1.0:
                        rank = row["rank"]
                    else:
                        # Fallback for test: normalize index
                        rank = group.index.get_loc(row.name) / len(group)

                cells.append((cid, rank))

            # Sort by rank
            cells.sort(key=lambda x: x[1])
            ordered_ids = [c[0] for c in cells]
            results[nb_id] = ordered_ids

        return results

    # Reconstruct orders for Validation set
    val_predictions_dict = reconstruct_order(df_val, df_feats_val, final_val_preds)

    # Compute Metric
    # Use distinct notebooks for evaluation to avoid weighted average by cell count
    df_val_nb = df_val.drop_duplicates(subset=["id"])
    score = compute_score(df_val_nb, val_predictions_dict)
    print(f"Validation Kendall Tau: {score:.4f}")

    assert -1.0 <= score <= 1.0, "Score out of valid range."

    # --------------------------------------------------------------------------
    # 8. Submission Generation
    # --------------------------------------------------------------------------
    print("\n--- Generating Submission ---")

    # Reconstruct orders for Test set
    test_predictions_dict = reconstruct_order(df_test, df_feats_test, final_test_preds)

    # Create DataFrame
    submission_rows = []
    for nb_id, cell_order in test_predictions_dict.items():
        submission_rows.append({"id": nb_id, "cell_order": " ".join(cell_order)})

    df_submission = pd.DataFrame(submission_rows)
    print(df_submission.head())

    # Save (Mock)
    sub_path = os.path.join(Config.WORKING_DIR, "submission", "submission.csv")
    df_submission.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()

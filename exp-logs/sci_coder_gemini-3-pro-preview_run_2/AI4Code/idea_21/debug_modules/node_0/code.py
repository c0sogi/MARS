import os
import pandas as pd
import numpy as np
import random
import shutil

# Import library components
from library.config import Config
from library.data_manager import DataManager
from library.vectorizer import DualVectorizer
from library.anchor_engine import AnchorExtractor
from library.stage1_ridge import RidgeStacker
from library.stage2_lgbm import LGBMRanker
from library.utils import kendall_tau_metric, convert_ranks_to_order

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)

if __name__ == "__main__":
    print("=== Starting Notebook Cell Ordering Demo ===\n")

    # --------------------------------------------------------------------------
    # 1. Configuration & Optimization for Demo
    # --------------------------------------------------------------------------
    # Override Config to use a separate demo directory and faster settings
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce complexity for speed
    Config.TFIDF_PARAMS["max_features"] = 2000
    Config.SVD_PARAMS["n_components"] = 16
    Config.SVD_PARAMS["n_iter"] = 2
    Config.LGBM_PARAMS["n_estimators"] = 50
    Config.N_FOLDS_STAGE1 = 3

    # Re-run setup to create new directories
    Config.setup()

    # --------------------------------------------------------------------------
    # 2. Data Subsampling
    # --------------------------------------------------------------------------
    print("Creating subsampled metadata for demonstration...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample subsets (Train: 200, Val: 50, Test: 50)
    sub_train = orig_train_meta.sample(n=200, random_state=42)
    sub_val = orig_val_meta.sample(n=50, random_state=42)
    sub_test = orig_test_meta.sample(n=50, random_state=42)

    # Save subsampled metadata
    meta_dir = os.path.join(DEMO_DIR, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    path_train = os.path.join(meta_dir, "train_metadata.csv")
    path_val = os.path.join(meta_dir, "val_metadata.csv")
    path_test = os.path.join(meta_dir, "test_metadata.csv")

    sub_train.to_csv(path_train, index=False)
    sub_val.to_csv(path_val, index=False)
    sub_test.to_csv(path_test, index=False)

    # Point Config to new metadata
    Config.TRAIN_METADATA_PATH = path_train
    Config.VAL_METADATA_PATH = path_val
    Config.TEST_METADATA_PATH = path_test

    # --------------------------------------------------------------------------
    # 3. Data Loading
    # --------------------------------------------------------------------------
    dm = DataManager()

    # Load all splits (load_cached_data=False ensures we process the new subsamples)
    df_train_all = dm.load_data("train", load_cached_data=False)
    df_val_all = dm.load_data("val", load_cached_data=False)
    df_test_all = dm.load_data("test", load_cached_data=False)

    print(
        f"Loaded Data Shapes: Train={df_train_all.shape}, Val={df_val_all.shape}, Test={df_test_all.shape}"
    )

    # --------------------------------------------------------------------------
    # 4. Vectorization (TF-IDF + SVD)
    # --------------------------------------------------------------------------
    vectorizer = DualVectorizer()

    # Fit on training text (Code + Markdown)
    print("Fitting vectorizers...")
    vectorizer.fit_or_load(df_train_all["source"], load_cached_data=False)

    # Transform all splits
    print("Transforming text data...")
    tfidf_train, svd_train = vectorizer.transform(df_train_all["source"])
    tfidf_val, svd_val = vectorizer.transform(df_val_all["source"])
    tfidf_test, svd_test = vectorizer.transform(df_test_all["source"])

    # --------------------------------------------------------------------------
    # 5. Feature Engineering: Anchor Extraction
    # --------------------------------------------------------------------------
    extractor = AnchorExtractor()

    # Extract features (returns DataFrame for Markdown cells)
    anchors_train = extractor.extract_features(
        df_train_all, tfidf_train, svd_train, "train", load_cached_data=False
    )
    anchors_val = extractor.extract_features(
        df_val_all, tfidf_val, svd_val, "val", load_cached_data=False
    )
    anchors_test = extractor.extract_features(
        df_test_all, tfidf_test, svd_test, "test", load_cached_data=False
    )

    # --------------------------------------------------------------------------
    # 6. Stage 1: Ridge Regression (Lexical Model)
    # --------------------------------------------------------------------------
    ridge = RidgeStacker()

    # Identify Markdown cells in Train/Val/Test
    # We only train/predict ranks for Markdown cells; Code cells are fixed anchors.
    mask_train_md = df_train_all["cell_type"] == "markdown"
    mask_val_md = df_val_all["cell_type"] == "markdown"
    mask_test_md = df_test_all["cell_type"] == "markdown"

    # Prepare Stage 1 Inputs
    X_train_md_tfidf = tfidf_train[mask_train_md]
    y_train_md = df_train_all.loc[mask_train_md, "pct_rank"].values
    groups_train = df_train_all.loc[mask_train_md, "ancestor_id"].values

    X_val_md_tfidf = tfidf_val[mask_val_md]
    X_test_md_tfidf = tfidf_test[mask_test_md]

    # Train & OOF
    print("Running Stage 1 (Ridge)...")
    s1_oof_train = ridge.fit_predict_oof(
        X_train_md_tfidf, y_train_md, groups_train, load_cached_data=False
    )

    # Inference
    s1_pred_val = ridge.predict(X_val_md_tfidf)
    s1_pred_test = ridge.predict(X_test_md_tfidf)

    # --------------------------------------------------------------------------
    # 7. Data Assembly for Stage 2
    # --------------------------------------------------------------------------
    def assemble_stage2_data(df_all, mask_md, anchor_features, s1_preds):
        """Helper to merge features for Stage 2"""
        # Get base Markdown DataFrame
        df_md = df_all[mask_md].copy().reset_index(drop=True)

        # Add Stage 1 Predictions
        df_md["s1_pred"] = s1_preds

        # Merge Anchor Features
        # Ensure 'cell_id' is the key.
        # Note: anchor_features comes from AnchorExtractor which iterates groups.
        df_md = df_md.merge(anchor_features, on="cell_id", how="left")

        # Select features for LGBM (exclude ID, strings, target)
        drop_cols = [
            "id",
            "cell_id",
            "cell_type",
            "source",
            "ancestor_id",
            "parent_id",
            "rank",
            "pct_rank",
        ]
        features = df_md.drop(columns=[c for c in drop_cols if c in df_md.columns])

        # Fill NaNs if any (e.g. from missing anchors)
        features = features.fillna(0.5)

        return features, df_md

    print("Assembling Stage 2 datasets...")
    X_train_s2, df_train_md_s2 = assemble_stage2_data(
        df_train_all, mask_train_md, anchors_train, s1_oof_train
    )
    y_train_s2 = df_train_md_s2["pct_rank"].values

    X_val_s2, df_val_md_s2 = assemble_stage2_data(
        df_val_all, mask_val_md, anchors_val, s1_pred_val
    )
    y_val_s2 = df_val_md_s2["pct_rank"].values

    X_test_s2, df_test_md_s2 = assemble_stage2_data(
        df_test_all, mask_test_md, anchors_test, s1_pred_test
    )

    # --------------------------------------------------------------------------
    # 8. Stage 2: LightGBM (Hybrid Ranker)
    # --------------------------------------------------------------------------
    lgbm = LGBMRanker()

    print("Training Stage 2 (LightGBM)...")
    lgbm.fit(
        X_train_s2, y_train_s2, X_val=X_val_s2, y_val=y_val_s2, load_cached_model=False
    )

    print("Predicting Test Ranks...")
    test_pred_ranks = lgbm.predict(X_test_s2)

    # Assign predictions back to DataFrame
    df_test_md_s2["pred_rank"] = test_pred_ranks

    # --------------------------------------------------------------------------
    # 9. Submission Generation
    # --------------------------------------------------------------------------
    print("Generating Submission...")

    submission_rows = []

    # We need to reconstruct the order for each test notebook
    # Group by notebook ID
    # Note: We need the code cells for the test set too
    df_test_code = df_test_all[df_test_all["cell_type"] != "markdown"]

    test_ids = df_test_all["id"].unique()

    for nb_id in test_ids:
        # Get Code Cells
        code_cells = df_test_code[df_test_code["id"] == nb_id]["cell_id"].tolist()

        # Get Markdown Cells and their predicted ranks
        md_rows = df_test_md_s2[df_test_md_s2["id"] == nb_id]
        md_cells = md_rows["cell_id"].tolist()
        md_ranks = md_rows["pred_rank"].tolist()

        # Convert to order string
        order_str = convert_ranks_to_order(code_cells, md_cells, md_ranks)

        submission_rows.append({"id": nb_id, "cell_order": order_str})

    df_submission = pd.DataFrame(submission_rows)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # --------------------------------------------------------------------------
    # 10. Validation & Metrics
    # --------------------------------------------------------------------------
    print("\n=== Validation ===")

    # 1. Validate Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created."
    assert (
        len(df_submission) == 50
    ), f"Expected 50 rows in submission, got {len(df_submission)}"

    # 2. Calculate Metric on Validation Set
    # We need to generate 'cell_order' predictions for the validation set to compare with ground truth
    print("Calculating Kendall Tau on Validation Set...")

    # Predict on Val
    val_pred_ranks = lgbm.predict(X_val_s2)
    df_val_md_s2["pred_rank"] = val_pred_ranks

    val_rows = []
    df_val_code = df_val_all[df_val_all["cell_type"] != "markdown"]
    val_ids = df_val_all["id"].unique()

    for nb_id in val_ids:
        code_cells = df_val_code[df_val_code["id"] == nb_id]["cell_id"].tolist()
        md_rows = df_val_md_s2[df_val_md_s2["id"] == nb_id]
        md_cells = md_rows["cell_id"].tolist()
        md_ranks = md_rows["pred_rank"].tolist()

        order_str = convert_ranks_to_order(code_cells, md_cells, md_ranks)
        val_rows.append({"id": nb_id, "cell_order": order_str})

    df_val_pred = pd.DataFrame(val_rows)

    # Get Ground Truth (from metadata)
    df_val_true = sub_val[["id", "cell_order"]]

    # Compute Metric
    score = kendall_tau_metric(df_val_true, df_val_pred)
    print(f"Validation Kendall Tau: {score:.4f}")

    # Assert Score Validity
    assert -1.0 <= score <= 1.0, "Kendall Tau score out of range."

    print("\nDemo completed successfully.")

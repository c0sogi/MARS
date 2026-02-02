import os
import sys
import pandas as pd
import numpy as np
import random
import shutil
import warnings
import lightgbm as lgb
from sklearn.exceptions import ConvergenceWarning

# Import from the provided library
from library.config import Config
from library.utils import kendall_tau_metric
from library.dataset import NotebookLoader
from library.vectorization import TextPipeline
from library.feature_engineering import FeatureExtractor
from library.modeling import Stage1Ridge, Stage2LGBM
from library.pipeline import RankingPipeline


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def main():
    print("=== AI4Code Pipeline Demonstration ===")

    # --------------------------------------------------------------------------
    # 0. Setup and Configuration Overrides
    # --------------------------------------------------------------------------
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    set_seed(42)

    # Define temporary working directory for this demo
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    print(f"Working directory: {DEMO_DIR}")

    # --- Runtime Config Modification ---
    # We modify Config attributes directly to point to our demo paths and
    # use lightweight hyperparameters for speed.

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Paths for models/data
    Config.TFIDF_MODEL_PATH = os.path.join(DEMO_DIR, "text_vectorizer_tfidf.joblib")
    Config.SVD_MODEL_PATH = os.path.join(DEMO_DIR, "text_vectorizer_svd.joblib")
    Config.RIDGE_MODEL_PATH = os.path.join(DEMO_DIR, "stage1_ridge_model.joblib")
    Config.LGBM_MODEL_PATH = os.path.join(
        DEMO_DIR, "stage2_lgbm_model.joblib"
    )  # Using joblib extension for consistency in demo

    Config.TRAIN_FEATURES_PATH = os.path.join(DEMO_DIR, "train_features.parquet")
    Config.VAL_FEATURES_PATH = os.path.join(DEMO_DIR, "val_features.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(DEMO_DIR, "test_features.parquet")
    Config.STAGE1_OOF_PATH = os.path.join(
        DEMO_DIR, "stage1_oof_preds.npy"
    )  # Note: Demo uses npy/parquet depending on class impl

    # Reduce Complexity for Speed
    Config.VOCAB_SIZE = 1000
    Config.SVD_COMPONENTS = 10
    Config.CONTENT_PROJECTION_DIMS = 4
    Config.NEIGHBOR_K = 3

    # Update TF-IDF Params
    Config.TFIDF_PARAMS["max_features"] = Config.VOCAB_SIZE

    # Update SVD Params
    Config.SVD_PARAMS["n_components"] = Config.SVD_COMPONENTS
    Config.SVD_PARAMS["n_iter"] = 2

    # Update LGBM Params
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.EARLY_STOPPING_ROUNDS = 5
    Config.VERBOSE_EVAL = False

    # --------------------------------------------------------------------------
    # 1. Create Mini-Datasets (Subsampling Metadata)
    # --------------------------------------------------------------------------
    print("\n[1] Creating mini-datasets for rapid demonstration...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample subsets (ensure we have enough for 5-fold CV in Stage 1)
    # We need at least 5 groups for GroupKFold if n_splits=5
    n_train = 50
    n_val = 20
    n_test = 20

    mini_train_meta = orig_train_meta.iloc[:n_train].copy()
    mini_val_meta = orig_val_meta.iloc[:n_val].copy()
    mini_test_meta = orig_test_meta.iloc[:n_test].copy()

    # Save mini metadata
    mini_train_path = os.path.join(DEMO_DIR, "metadata", "train_metadata.csv")
    mini_val_path = os.path.join(DEMO_DIR, "metadata", "val_metadata.csv")
    mini_test_path = os.path.join(DEMO_DIR, "metadata", "test_metadata.csv")

    os.makedirs(os.path.dirname(mini_train_path), exist_ok=True)
    mini_train_meta.to_csv(mini_train_path, index=False)
    mini_val_meta.to_csv(mini_val_path, index=False)
    mini_test_meta.to_csv(mini_test_path, index=False)

    # Point Config to mini metadata
    Config.TRAIN_METADATA_PATH = mini_train_path
    Config.VAL_METADATA_PATH = mini_val_path
    Config.TEST_METADATA_PATH = mini_test_path

    print(f"    Train subset: {len(mini_train_meta)} notebooks")
    print(f"    Val subset:   {len(mini_val_meta)} notebooks")

    # --------------------------------------------------------------------------
    # 2. Notebook Loading
    # --------------------------------------------------------------------------
    print("\n[2] Loading Notebook Data...")
    loader = NotebookLoader()

    # Force reload (ignore cache logic for demo correctness)
    df_train = loader.load_train_data(load_cached_data=False)
    df_val = loader.load_val_data(load_cached_data=False)
    df_test = loader.load_test_data(load_cached_data=False)

    # Validation
    assert not df_train.empty, "Train dataframe is empty"
    assert "pct_rank" in df_train.columns, "Train dataframe missing target 'pct_rank'"
    assert "ancestor_id" in df_train.columns, "Train dataframe missing 'ancestor_id'"
    print(f"    Loaded {len(df_train)} training cells.")

    # --------------------------------------------------------------------------
    # 3. Text Vectorization (TF-IDF + SVD)
    # --------------------------------------------------------------------------
    print("\n[3] Fitting Text Pipeline...")
    text_pipeline = TextPipeline()

    # Fit on training markdown
    text_pipeline.fit_transform_corpus(df_train, load_cached_model=False)

    assert text_pipeline.is_fitted, "TextPipeline failed to fit."

    # Check transformation shape
    sample_text = df_train["source"].iloc[:5]
    svd_vecs = text_pipeline.transform_cells(sample_text)
    assert svd_vecs.shape == (
        5,
        Config.SVD_COMPONENTS,
    ), f"SVD output shape mismatch. Expected (5, {Config.SVD_COMPONENTS}), got {svd_vecs.shape}"
    print("    Text Pipeline fitted and verified.")

    # --------------------------------------------------------------------------
    # 4. Stage 1: Ridge Regression (OOF & Predict)
    # --------------------------------------------------------------------------
    print("\n[4] Running Stage 1 (Ridge Regression)...")
    stage1 = Stage1Ridge()

    # Generate OOF for Train
    # Note: With very few groups, GroupKFold might warn or fail if n_splits > n_groups.
    # Our sample size is 50 notebooks, so 50 groups. 5 splits is fine.
    train_s1_preds = stage1.get_oof_predictions(
        df_train, text_pipeline, load_cached_data=False
    )

    # Validation
    # OOF should only contain markdown cells
    n_md_train = df_train[df_train["cell_type"] == "markdown"].shape[0]
    assert (
        len(train_s1_preds) == n_md_train
    ), f"Stage 1 OOF count mismatch. Expected {n_md_train}, got {len(train_s1_preds)}"
    assert "stage1_pred" in train_s1_preds.columns

    # Predict on Val
    val_s1_preds = stage1.predict(df_val, text_pipeline)
    assert not val_s1_preds.empty

    print("    Stage 1 OOF and Val predictions generated.")

    # --------------------------------------------------------------------------
    # 5. Feature Engineering (Content-Aware Neighbors)
    # --------------------------------------------------------------------------
    print("\n[5] Extracting Features...")
    extractor = FeatureExtractor()

    # Extract features for Train
    train_features = extractor.extract_features(
        df_train, text_pipeline, mode="train", load_cached_data=False
    )

    # Extract features for Val
    val_features = extractor.extract_features(
        df_val, text_pipeline, mode="val", load_cached_data=False
    )

    # Validation
    # Features should include neighbor_rank_mean, neighbor_content_0, etc.
    expected_col = f"neighbor_content_{Config.CONTENT_PROJECTION_DIMS - 1}"
    assert (
        expected_col in train_features.columns
    ), f"Missing feature column {expected_col}"
    assert "target_rank" in train_features.columns, "Missing target in train features"

    print(f"    Extracted features shape: {train_features.shape}")

    # --------------------------------------------------------------------------
    # 6. Stage 2: LightGBM Stacking
    # --------------------------------------------------------------------------
    print("\n[6] Training Stage 2 (LightGBM)...")
    stage2 = Stage2LGBM()

    # Train
    model = stage2.train(train_features, val_features, train_s1_preds, val_s1_preds)

    # Predict on Val
    val_final_preds = stage2.predict(val_features, val_s1_preds)

    assert "pred_rank" in val_final_preds.columns
    assert len(val_final_preds) == len(val_features), "Prediction count mismatch"

    print("    Stage 2 training and prediction complete.")

    # --------------------------------------------------------------------------
    # 7. Evaluation (Kendall Tau)
    # --------------------------------------------------------------------------
    print("\n[7] Evaluating Validation Performance...")

    # Prepare Ground Truth for Validation
    # We need a dataframe with 'id' and 'cell_order'
    val_gt_rows = []
    for nb_id, group in df_val.groupby("id"):
        # Sort by rank to get correct order
        sorted_cells = group.sort_values("rank")["cell_id"].tolist()
        val_gt_rows.append({"id": nb_id, "cell_order": " ".join(sorted_cells)})
    df_val_gt = pd.DataFrame(val_gt_rows)

    # Prepare Predictions Dictionary
    # We need to merge predicted markdown ranks with known code ranks
    # 1. Get Code Ranks (Ground Truth for Val)
    code_mask = df_val["cell_type"] == "code"
    val_code = df_val[code_mask][["id", "cell_id", "pct_rank"]].copy()
    val_code.rename(columns={"pct_rank": "rank"}, inplace=True)

    # 2. Get Markdown Preds
    val_md = val_final_preds[["id", "cell_id", "pred_rank"]].copy()
    val_md.rename(columns={"pred_rank": "rank"}, inplace=True)

    # 3. Combine
    val_all = pd.concat([val_code, val_md])

    preds_dict = {}
    for nb_id, group in val_all.groupby("id"):
        sorted_ids = group.sort_values("rank")["cell_id"].tolist()
        preds_dict[nb_id] = sorted_ids

    # Calculate Metric
    score = kendall_tau_metric(df_val_gt, preds_dict)
    print(f"    Validation Kendall Tau: {score:.4f}")

    assert 0.0 <= score <= 1.0, "Kendall Tau score out of bounds"

    # --------------------------------------------------------------------------
    # 8. Inference on Test Set
    # --------------------------------------------------------------------------
    print("\n[8] Running Inference on Test Set...")

    # 1. Stage 1 Predict
    test_s1_preds = stage1.predict(df_test, text_pipeline)

    # 2. Feature Extract
    test_features = extractor.extract_features(
        df_test, text_pipeline, mode="test", load_cached_data=False
    )

    # 3. Stage 2 Predict
    test_final_preds = stage2.predict(test_features, test_s1_preds)

    # 4. Generate Submission
    # We use the RankingPipeline helper for this part to demonstrate its usage
    # We instantiate a dummy pipeline just to access the method, or copy logic.
    # Since we are demonstrating classes, let's use the RankingPipeline class properly
    # but inject our pre-calculated data if possible.
    # However, RankingPipeline is designed to run end-to-end.
    # We will manually replicate the submission generation logic here for clarity
    # as we already have the predictions.

    print("    Generating submission file...")

    # Process Code Cells (Equidistant ranks for Test)
    code_mask = df_test["cell_type"] == "code"
    test_code = df_test[code_mask].copy()

    test_code_list = []
    for nb_id, group in test_code.groupby("id"):
        n = len(group)
        if n > 1:
            ranks = np.arange(n) / (n - 1.0)
        else:
            ranks = np.zeros(n)
        temp = group[["id", "cell_id"]].copy()
        temp["rank"] = ranks
        test_code_list.append(temp)

    if test_code_list:
        df_test_code = pd.concat(test_code_list)
    else:
        df_test_code = pd.DataFrame(columns=["id", "cell_id", "rank"])

    # Process Markdown
    df_test_md = test_final_preds[["id", "cell_id", "pred_rank"]].copy()
    df_test_md.rename(columns={"pred_rank": "rank"}, inplace=True)

    # Merge
    test_all = pd.concat([df_test_code, df_test_md])

    submission_rows = []
    for nb_id, group in test_all.groupby("id"):
        sorted_ids = group.sort_values("rank")["cell_id"].tolist()
        submission_rows.append({"id": nb_id, "cell_order": " ".join(sorted_ids)})

    df_submission = pd.DataFrame(submission_rows)

    # Save
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    df_submission.to_csv(Config.SUBMISSION_PATH, index=False)

    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission rows: {len(df_submission)}")

    assert len(df_submission) == len(
        mini_test_meta
    ), f"Submission row count mismatch. Expected {len(mini_test_meta)}, got {len(df_submission)}"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()

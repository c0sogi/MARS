import os
import shutil
import pandas as pd
import numpy as np
import joblib
from library import (
    config,
    utils,
    data_factory,
    feature_engine,
    model_zoo,
    training_pipeline,
    inference_pipeline,
)


def main():
    # ==========================================================================
    # 1. Setup & Configuration Override
    # ==========================================================================
    print("\n[Demo] Setting up configuration for rapid testing...")

    # Define demo directories
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_SUBMISSION_DIR = "./working/demo_submission"

    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR)
    os.makedirs(DEMO_SUBMISSION_DIR, exist_ok=True)

    # Override config paths to use demo directories
    config.WORKING_DIR = DEMO_WORKING_DIR
    config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    config.CACHE_TRAIN_DATAFRAME = os.path.join(
        DEMO_WORKING_DIR, "train_dataframe.parquet"
    )
    config.CACHE_VAL_DATAFRAME = os.path.join(DEMO_WORKING_DIR, "val_dataframe.parquet")
    config.CACHE_TEST_DATAFRAME = os.path.join(
        DEMO_WORKING_DIR, "test_dataframe.parquet"
    )

    config.CACHE_TFIDF_VECTORIZER = os.path.join(
        DEMO_WORKING_DIR, "tfidf_vectorizer.joblib"
    )
    config.CACHE_SVD_MODEL = os.path.join(DEMO_WORKING_DIR, "svd_model.joblib")
    config.CACHE_RIDGE_MODEL = os.path.join(DEMO_WORKING_DIR, "ridge_model.joblib")

    config.CACHE_TRAIN_FEATURES = os.path.join(
        DEMO_WORKING_DIR, "train_features.parquet"
    )
    config.CACHE_VAL_FEATURES = os.path.join(DEMO_WORKING_DIR, "val_features.parquet")
    config.CACHE_TEST_FEATURES = os.path.join(DEMO_WORKING_DIR, "test_features.parquet")

    config.CACHE_STAGE1_OOF = os.path.join(DEMO_WORKING_DIR, "stage1_oof_preds.parquet")
    config.CACHE_STAGE1_TEST = os.path.join(
        DEMO_WORKING_DIR, "stage1_test_preds.parquet"
    )

    # Override Hyperparameters for Speed
    # Reduce SVD components to fit small data subset
    config.SVD_COMPONENTS = 5
    config.SVD_PARAMS["n_components"] = 5
    config.SVD_PARAMS["n_iter"] = 2

    # Reduce TF-IDF vocabulary size
    config.TFIDF_PARAMS["max_features"] = 100
    config.TFIDF_PARAMS["min_df"] = 1  # Allow rare words in small subset

    # Reduce LightGBM complexity
    config.LGBM_PARAMS["n_estimators"] = 10
    config.LGBM_PARAMS["num_leaves"] = 5
    config.EARLY_STOPPING_ROUNDS = 5

    # Set seed
    utils.set_seed(42)

    # ==========================================================================
    # 2. Prepare Data Subsets
    # ==========================================================================
    print("[Demo] Creating data subsets (20 notebooks each)...")

    # Load original metadata
    orig_train_meta = pd.read_csv(config.TRAIN_METADATA_PATH)
    orig_val_meta = pd.read_csv(config.VAL_METADATA_PATH)
    orig_test_meta = pd.read_csv(config.TEST_METADATA_PATH)

    # Sample 20 notebooks
    # Ensure we have enough groups for GroupKFold (5 splits)
    subset_train = orig_train_meta.sample(n=20, random_state=42)
    subset_val = orig_val_meta.sample(n=20, random_state=42)
    subset_test = orig_test_meta.sample(n=20, random_state=42)

    # Save subsets to demo working dir
    demo_train_meta_path = os.path.join(DEMO_WORKING_DIR, "train_metadata.csv")
    demo_val_meta_path = os.path.join(DEMO_WORKING_DIR, "val_metadata.csv")
    demo_test_meta_path = os.path.join(DEMO_WORKING_DIR, "test_metadata.csv")

    subset_train.to_csv(demo_train_meta_path, index=False)
    subset_val.to_csv(demo_val_meta_path, index=False)
    subset_test.to_csv(demo_test_meta_path, index=False)

    # Point config to these new metadata files
    config.TRAIN_METADATA_PATH = demo_train_meta_path
    config.VAL_METADATA_PATH = demo_val_meta_path
    config.TEST_METADATA_PATH = demo_test_meta_path

    # ==========================================================================
    # 3. Verify Metric Logic
    # ==========================================================================
    print("[Demo] Verifying Kendall Tau Metric...")
    # Case: Perfect match
    df_true = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    df_pred = pd.DataFrame({"id": ["nb1"], "cell_order": ["a b c"]})
    score = utils.kendall_tau_metric(df_true, df_pred)
    assert score == 1.0, f"Expected 1.0, got {score}"

    # Case: Complete inversion (a b c -> c b a)
    # Pairs: (a,b), (a,c), (b,c). Inversions: (c,b), (c,a), (b,a) -> 3 inversions.
    # Total pairs: 3. Score: 1 - 4 * (3/3) = -3.0? No, formula is 1 - 4 * (S / (n(n-1)))
    # n=3, n(n-1)=6. S=3. Score = 1 - 4 * (3/6) = 1 - 2 = -1.0. Correct.
    df_pred_inv = pd.DataFrame({"id": ["nb1"], "cell_order": ["c b a"]})
    score_inv = utils.kendall_tau_metric(df_true, df_pred_inv)
    assert abs(score_inv - (-1.0)) < 1e-6, f"Expected -1.0, got {score_inv}"
    print("Metric check passed.")

    # ==========================================================================
    # 4. Verify Data Loading (Data Factory)
    # ==========================================================================
    print("[Demo] Testing Data Factory...")
    df_train = data_factory.load_train_data(load_cached_data=False)
    print(f"Loaded {len(df_train)} training cells.")

    expected_cols = [
        "id",
        "cell_id",
        "cell_type",
        "source",
        "ancestor_id",
        "rank",
        "is_code",
    ]
    for col in expected_cols:
        assert col in df_train.columns, f"Missing column {col} in train data"

    # Check that rank is normalized for markdown
    md_ranks = df_train[df_train["is_code"] == 0]["rank"]
    if len(md_ranks) > 0:
        assert md_ranks.min() >= 0.0, "Rank should be >= 0"
        # Rank can be > 1.0 slightly due to interpolation logic (1 + 1/N), but generally around [0,1]
        assert md_ranks.max() <= 2.0, "Rank should be reasonably normalized"

    # ==========================================================================
    # 5. Verify Feature Engineering
    # ==========================================================================
    print("[Demo] Testing Feature Engine...")
    # This will fit TF-IDF and SVD on the small train subset
    df_train_feats = feature_engine.generate_features(
        mode="train", load_cached_data=False
    )

    # Check feature columns
    feat_cols = ["lex_mean_rank", "lat_mean_rank", "svd_0", "n_code_cells"]
    for col in feat_cols:
        assert col in df_train_feats.columns, f"Missing feature {col}"

    # Check SVD dimensions
    svd_cols = [c for c in df_train_feats.columns if c.startswith("svd_")]
    assert (
        len(svd_cols) == config.SVD_COMPONENTS
    ), f"Expected {config.SVD_COMPONENTS} SVD cols, got {len(svd_cols)}"

    # Check cache existence
    assert os.path.exists(config.CACHE_TFIDF_VECTORIZER), "TF-IDF vectorizer not saved"
    assert os.path.exists(config.CACHE_SVD_MODEL), "SVD model not saved"
    assert os.path.exists(config.CACHE_TRAIN_FEATURES), "Train features not saved"

    # ==========================================================================
    # 6. Verify Training Pipeline
    # ==========================================================================
    print("[Demo] Running Training Pipeline...")
    # We already generated train features, pipeline will load them or regenerate.
    # We let the pipeline handle validation feature generation.
    training_pipeline.run_training_pipeline(load_cached_data=True)

    # Check artifacts
    assert os.path.exists(config.CACHE_RIDGE_MODEL), "Ridge model not saved"
    assert os.path.exists(
        os.path.join(config.WORKING_DIR, "lgbm_model.txt")
    ), "LGBM model not saved"
    assert os.path.exists(config.CACHE_STAGE1_OOF), "OOF preds not saved"

    # ==========================================================================
    # 7. Verify Inference Pipeline
    # ==========================================================================
    print("[Demo] Running Inference Pipeline...")
    inference_pipeline.run_inference_pipeline(load_cached_data=False)

    # Check submission
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found"

    df_sub = pd.read_csv(config.SUBMISSION_PATH)
    print(f"Submission shape: {df_sub.shape}")
    assert (
        df_sub.shape[0] == 20
    ), f"Expected 20 rows in submission, got {df_sub.shape[0]}"
    assert "id" in df_sub.columns and "cell_order" in df_sub.columns

    # Verify content format (space delimited ids)
    sample_order = df_sub.iloc[0]["cell_order"]
    assert isinstance(sample_order, str)
    assert len(sample_order.split()) > 0

    print("\n[Demo] All tests passed successfully!")


if __name__ == "__main__":
    main()

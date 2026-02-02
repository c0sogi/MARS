import os
import pandas as pd
import numpy as np
import shutil
from sklearn.metrics import mean_absolute_error

# Import library components
from library.config import Config
from library.utils import set_seed, compute_kendall_tau, load_joblib
from library.train_pipeline import TrainPipeline
from library.inference_pipeline import InferencePipeline


def create_subset_metadata(source_path, dest_path, n_samples):
    """
    Helper to create a small subset of metadata for demonstration purposes.
    """
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Source metadata not found: {source_path}")

    df = pd.read_csv(source_path)
    # Sample subset, ensuring we don't exceed available rows
    n = min(n_samples, len(df))
    df_subset = df.sample(n=n, random_state=42).reset_index(drop=True)

    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    df_subset.to_csv(dest_path, index=False)
    print(f"Created subset metadata at {dest_path} with {len(df_subset)} rows.")
    return df_subset


def run_demo():
    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("Initializing Demo Configuration...")

    # Define demo directories
    DEMO_WORKING_DIR = "./working/demo_run"
    DEMO_METADATA_DIR = os.path.join(DEMO_WORKING_DIR, "metadata")
    DEMO_SUBMISSION_DIR = os.path.join(DEMO_WORKING_DIR, "submission")

    # Clean up previous demo run if exists
    if os.path.exists(DEMO_WORKING_DIR):
        shutil.rmtree(DEMO_WORKING_DIR)
    os.makedirs(DEMO_WORKING_DIR, exist_ok=True)

    # Override Config attributes globally
    # This ensures all library classes use these new paths and parameters
    Config.WORKING_DIR = DEMO_WORKING_DIR
    Config.SUBMISSION_DIR = DEMO_SUBMISSION_DIR
    Config.SUBMISSION_PATH = os.path.join(DEMO_SUBMISSION_DIR, "submission.csv")

    # Update Metadata Paths to point to our future subsets
    Config.TRAIN_METADATA_PATH = os.path.join(DEMO_METADATA_DIR, "train_metadata.csv")
    Config.VAL_METADATA_PATH = os.path.join(DEMO_METADATA_DIR, "val_metadata.csv")
    Config.TEST_METADATA_PATH = os.path.join(DEMO_METADATA_DIR, "test_metadata.csv")

    # Update Cache Paths (Models/Features) to reside in demo working dir
    Config.CACHE_TFIDF_MODEL = os.path.join(DEMO_WORKING_DIR, "tfidf_vectorizer.joblib")
    Config.CACHE_SVD_MODEL = os.path.join(DEMO_WORKING_DIR, "svd_model.joblib")
    Config.CACHE_RIDGE_MODEL = os.path.join(
        DEMO_WORKING_DIR, "stage1_ridge_model.joblib"
    )
    Config.CACHE_STAGE1_OOF = os.path.join(DEMO_WORKING_DIR, "stage1_oof_preds.parquet")
    Config.CACHE_TRAIN_FEATURES = os.path.join(
        DEMO_WORKING_DIR, "demo_train_stage2_features.parquet"
    )
    Config.CACHE_VAL_FEATURES = os.path.join(
        DEMO_WORKING_DIR, "demo_val_stage2_features.parquet"
    )
    Config.CACHE_TEST_FEATURES = os.path.join(
        DEMO_WORKING_DIR, "demo_test_stage2_features.parquet"
    )
    Config.CACHE_LGBM_MODEL = os.path.join(DEMO_WORKING_DIR, "stage2_lgbm_model.txt")

    # Optimize Hyperparameters for Speed
    Config.LGBM_PARAMS["n_estimators"] = 10  # Very few trees for demo
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.SVD_N_COMPONENTS = 16  # Reduce dimensionality for speed
    Config.TFIDF_PARAMS["max_features"] = 1000  # Reduce vocab size
    Config.N_FOLDS = 2  # Fewer folds for Stage 1 OOF

    set_seed(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Prepare Data Subsets
    # --------------------------------------------------------------------------
    print("\n--- Step 1: Creating Data Subsets ---")
    # We use the original metadata files provided in ./metadata to create small subsets
    orig_train_meta = "./metadata/train_metadata.csv"
    orig_val_meta = "./metadata/val_metadata.csv"
    orig_test_meta = "./metadata/test_metadata.csv"

    df_train_sub = create_subset_metadata(
        orig_train_meta, Config.TRAIN_METADATA_PATH, n_samples=50
    )
    df_val_sub = create_subset_metadata(
        orig_val_meta, Config.VAL_METADATA_PATH, n_samples=20
    )
    df_test_sub = create_subset_metadata(
        orig_test_meta, Config.TEST_METADATA_PATH, n_samples=20
    )

    # --------------------------------------------------------------------------
    # 3. Execute Training Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Running Training Pipeline ---")
    trainer = TrainPipeline()

    # Run training (load_cached_data=False to force execution)
    trainer.run(load_cached_data=False)

    # Verify Artifacts
    assert os.path.exists(Config.CACHE_TFIDF_MODEL), "TF-IDF model not saved."
    assert os.path.exists(Config.CACHE_RIDGE_MODEL), "Stage 1 Ridge model not saved."
    assert os.path.exists(Config.CACHE_LGBM_MODEL), "Stage 2 LGBM model not saved."
    assert os.path.exists(Config.CACHE_TRAIN_FEATURES), "Train features not saved."

    print("Training artifacts verified successfully.")

    # --------------------------------------------------------------------------
    # 4. Execute Inference Pipeline
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Running Inference Pipeline ---")
    inferencer = InferencePipeline()

    # Run inference
    inferencer.predict_test_set(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created."

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")

    assert len(df_sub) == len(
        df_test_sub
    ), f"Submission row count mismatch. Expected {len(df_test_sub)}, got {len(df_sub)}"
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns are incorrect."

    # Check one example
    example_order = df_sub.iloc[0]["cell_order"]
    assert (
        isinstance(example_order, str) and len(example_order) > 0
    ), "Submission cell_order is empty or invalid."

    # --------------------------------------------------------------------------
    # 5. Manual Validation Logic Check
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Verifying Logic with Validation Set ---")
    # We will manually load the validation features and models to check metric calculation

    # Load processed validation features
    df_val_feats = pd.read_parquet(Config.CACHE_VAL_FEATURES)

    # Load Stage 1 predictions for validation (computed during pipeline)
    # Note: In the pipeline, these are computed on the fly.
    # To verify logic, we'll re-predict using Stage 1 model.

    # Load vectorizer and model
    tfidf = load_joblib(Config.CACHE_TFIDF_MODEL)
    ridge = load_joblib(Config.CACHE_RIDGE_MODEL)

    # Load raw validation data to get text
    from library.data_processing import NotebookProcessor

    processor = NotebookProcessor()
    df_val_raw = processor.load_val_data(
        load_cached_data=True
    )  # Should load from cache created by pipeline
    df_val_md = df_val_raw[df_val_raw["cell_type"] == "markdown"].reset_index(drop=True)

    # Vectorize and Predict Stage 1
    val_text = df_val_md["source"].astype(str).fillna("")
    X_val_sparse = tfidf.transform(val_text)
    preds_s1 = ridge.predict(X_val_sparse)

    # Check MAE of Stage 1
    mae_s1 = mean_absolute_error(df_val_md["pct_rank"], preds_s1)
    print(f"Manual Check - Stage 1 MAE: {mae_s1:.4f}")

    # Check Kendall Tau using the utility function
    # We need to construct a 'predicted' dataframe format
    # For this check, let's pretend our 'predicted order' is based solely on Stage 1 ranks

    # Create a prediction dataframe for validation set
    val_submission_rows = []
    df_val_raw["pred_rank"] = 0.0  # Default

    # Assign code ranks (fixed) and md ranks (predicted)
    # Code ranks in val are their ground truth pct_rank (or equidistant if we were doing pure inference)
    # For metric check against ground truth, we want to see how well our sorting works.

    # Map predictions back to raw dataframe
    # We iterate carefully to match indices
    md_indices = df_val_raw[df_val_raw["cell_type"] == "markdown"].index
    df_val_raw.loc[md_indices, "pred_rank"] = preds_s1

    # For code cells, let's use their actual rank to isolate markdown sorting performance
    code_indices = df_val_raw[df_val_raw["cell_type"] == "code"].index
    df_val_raw.loc[code_indices, "pred_rank"] = df_val_raw.loc[code_indices, "pct_rank"]

    for nb_id, group in df_val_raw.groupby("id"):
        sorted_group = group.sort_values("pred_rank")
        cell_order = " ".join(sorted_group["cell_id"].astype(str).tolist())
        val_submission_rows.append({"id": nb_id, "cell_order": cell_order})

    df_val_pred = pd.DataFrame(val_submission_rows)

    # Ground Truth DataFrame
    df_val_gt = df_val_sub[["id", "cell_order"]]

    # Compute Metric
    kt_score = compute_kendall_tau(df_val_gt, df_val_pred)
    print(f"Manual Check - Validation Kendall Tau (Stage 1 only): {kt_score:.4f}")

    assert 0.0 <= kt_score <= 1.0, "Kendall Tau score out of bounds."

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

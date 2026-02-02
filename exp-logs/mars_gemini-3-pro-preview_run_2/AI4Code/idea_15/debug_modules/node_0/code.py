import os
import sys
import random
import numpy as np
import pandas as pd
import warnings
import shutil
from sklearn.metrics import mean_absolute_error

# Import library components
from library.config import Config
from library.data_manager import DataManager
from library.feature_engine import FeatureEngine
from library.stage1_model import RidgeStacker
from library.stage2_model import LGBMRanker
from library.inference import SubmissionGenerator
from library.utils import kendall_tau, get_ranks


# ------------------------------------------------------------------------------
# 1. Setup and Configuration Overrides
# ------------------------------------------------------------------------------
def setup_demo_environment():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    # Set seeds
    random.seed(42)
    np.random.seed(42)

    print(">>> Configuring Demo Environment...")

    # Modify Config for speed and resource usage
    Config.VOCAB_SIZE = 1000  # Reduced from 60000
    Config.SVD_COMPONENTS = 10  # Reduced from 128
    Config.SVD_N_ITER = 1  # Reduced from 5
    Config.LGBM_PARAMS["n_estimators"] = 10  # Reduced from 5000
    Config.LGBM_PARAMS["num_leaves"] = 10
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_run/submission"

    # Create directories
    Config.setup()

    # Define temporary metadata paths
    Config.TRAIN_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_train_meta.csv")
    Config.VAL_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_val_meta.csv")
    Config.TEST_METADATA_PATH = os.path.join(Config.WORKING_DIR, "mini_test_meta.csv")


def create_mini_datasets():
    print(">>> Creating Mini Datasets...")

    # Load original metadata
    orig_train = pd.read_csv("./metadata/train_metadata.csv")
    orig_val = pd.read_csv("./metadata/val_metadata.csv")
    orig_test = pd.read_csv("./metadata/test_metadata.csv")

    # Sample subsets
    # Ensure we have enough samples for GroupKFold (at least 5 groups)
    mini_train = orig_train.sample(n=50, random_state=42)
    mini_val = orig_val.sample(n=20, random_state=42)
    mini_test = orig_test.sample(n=20, random_state=42)

    # Save to working directory
    mini_train.to_csv(Config.TRAIN_METADATA_PATH, index=False)
    mini_val.to_csv(Config.VAL_METADATA_PATH, index=False)
    mini_test.to_csv(Config.TEST_METADATA_PATH, index=False)

    print(f"    Train samples: {len(mini_train)}")
    print(f"    Val samples:   {len(mini_val)}")
    print(f"    Test samples:  {len(mini_test)}")


# ------------------------------------------------------------------------------
# 2. Pipeline Execution
# ------------------------------------------------------------------------------
def run_pipeline_demo():
    # --- A. Data Loading ---
    print("\n>>> [Step A] Data Manager: Loading and Processing...")
    dm = DataManager()

    # We force load_cached_data=False to ensure processing logic runs
    df_train = dm.get_train_data(load_cached_data=False)
    df_val = dm.get_val_data(load_cached_data=False)

    # Validation
    assert not df_train.empty, "Training dataframe is empty"
    assert "source" in df_train.columns, "Missing 'source' column"
    assert "cell_type" in df_train.columns, "Missing 'cell_type' column"
    print(f"    Processed Train Data Shape: {df_train.shape}")
    print(f"    Processed Val Data Shape:   {df_val.shape}")

    # --- B. Feature Engineering ---
    print("\n>>> [Step B] Feature Engine: Fitting and Transforming...")
    fe = FeatureEngine()

    # Fit on training data
    fe.fit(df_train)

    # Transform
    # Returns sparse matrix (X_sparse) and dense dataframe (df_features)
    X_train_sparse, df_train_features = fe.transform(
        df_train, name="features_train", load_cached_data=False
    )
    X_val_sparse, df_val_features = fe.transform(
        df_val, name="features_val", load_cached_data=False
    )

    # Validation
    assert (
        X_train_sparse.shape[0] == df_train_features.shape[0]
    ), "Sparse/Dense row mismatch (Train)"
    assert (
        X_train_sparse.shape[1] == Config.VOCAB_SIZE
    ), f"Vocab size mismatch. Expected {Config.VOCAB_SIZE}"
    assert (
        f"svd_{Config.SVD_COMPONENTS - 1}" in df_train_features.columns
    ), "SVD columns missing"
    print(f"    Sparse Matrix Shape: {X_train_sparse.shape}")
    print(f"    Dense Features Shape: {df_train_features.shape}")

    # --- C. Stage 1: Ridge Regression ---
    print("\n>>> [Step C] Stage 1 Model: Ridge Stacker...")
    stage1 = RidgeStacker()

    # Prepare targets and groups
    # df_train_features contains only markdown cells (usually), we need to ensure alignment
    # The feature engine aligns output with the returned dataframe.
    y_train = df_train_features["rank"].values
    groups_train = df_train_features["ancestor_id"].values

    # Train OOF
    oof_preds = stage1.train_oof(
        X_train_sparse, y_train, groups_train, n_splits=3, load_cached_data=False
    )

    # Fit Final Model
    stage1.fit_final(X_train_sparse, y_train)

    # Predict on Val
    val_ridge_preds = stage1.predict(X_val_sparse)

    # Validation
    assert len(oof_preds) == len(y_train), "OOF preds length mismatch"
    assert len(val_ridge_preds) == len(df_val_features), "Val preds length mismatch"
    print(f"    Stage 1 OOF MAE: {mean_absolute_error(y_train, oof_preds):.4f}")

    # --- D. Stage 2: LightGBM ---
    print("\n>>> [Step D] Stage 2 Model: LGBM Ranker...")
    stage2 = LGBMRanker()

    # Add Stage 1 predictions as features
    df_train_features["pred_ridge"] = oof_preds
    df_val_features["pred_ridge"] = val_ridge_preds

    # Define feature columns
    feature_cols = [
        "lex_mean",
        "lex_std",
        "lat_mean",
        "lat_std",
        "sym_mean",
        "sym_std",
        "md_ratio",
        "total_code",
        "pred_ridge",
    ]
    feature_cols += [f"svd_{i}" for i in range(Config.SVD_COMPONENTS)]

    # Train
    stage2.train(df_train_features, df_val_features, feature_cols, target_col="rank")

    # Predict
    val_lgbm_preds = stage2.predict(df_val_features, feature_cols)

    # Validation
    assert len(val_lgbm_preds) == len(df_val_features), "Stage 2 pred length mismatch"
    print(
        f"    Stage 2 Val MAE: {mean_absolute_error(df_val_features['rank'], val_lgbm_preds):.4f}"
    )

    return df_val, df_val_features, val_lgbm_preds


# ------------------------------------------------------------------------------
# 3. Inference and Metric Verification
# ------------------------------------------------------------------------------
def verify_inference_and_metric(df_val_raw, df_val_features, val_preds):
    print("\n>>> [Step E] Verification: Reconstruction and Kendall Tau...")

    # Add predictions back to the features dataframe
    df_val_features["pred_rank"] = val_preds

    # We need to reconstruct the order for each notebook in the validation set
    # The validation set contains both code and markdown.
    # df_val_features only contains Markdown cells (processed by FeatureEngine).
    # We need to merge these predictions back with the code cells from df_val_raw.

    # 1. Get Code Cells with implicit rank
    df_code = df_val_raw[df_val_raw["cell_type"] == "code"].copy()

    # Calculate code ranks based on original position
    # In the raw data, code cells are in correct order.
    # We can just use their existing 'rank' if available, or recalculate.
    # The 'rank' column in df_val_raw comes from ground truth, so let's simulate
    # the inference scenario where we calculate it from position.
    code_counts = df_code.groupby("id")["cell_id"].transform("count")
    code_cumcounts = df_code.groupby("id").cumcount()
    df_code["pred_rank"] = np.where(
        code_counts > 1, code_cumcounts / (code_counts - 1), 0.0
    )

    # 2. Get Markdown Cells with predicted rank
    df_md = df_val_features[["id", "cell_id", "pred_rank"]].copy()

    # 3. Combine
    df_all = pd.concat(
        [df_md, df_code[["id", "cell_id", "pred_rank"]]], ignore_index=True
    )

    # 4. Sort to get predicted order
    df_all.sort_values(by=["id", "pred_rank"], inplace=True)
    predicted_orders = df_all.groupby("id")["cell_id"].apply(list).to_dict()

    # 5. Get Ground Truth Orders
    # We can extract this from the metadata file we created
    val_meta = pd.read_csv(Config.VAL_METADATA_PATH)
    gt_orders = {}
    for _, row in val_meta.iterrows():
        gt_orders[row["id"]] = row["cell_order"].split()

    # 6. Calculate Kendall Tau
    pred_list = []
    gt_list = []

    for nb_id, gt_order in gt_orders.items():
        if nb_id in predicted_orders:
            gt_list.append(gt_order)
            pred_list.append(predicted_orders[nb_id])

    score = kendall_tau(gt_list, pred_list)
    print(f"    Validation Kendall Tau Score: {score:.4f}")

    # Basic sanity check: Score should be between -1 and 1
    assert -1.0 <= score <= 1.0, "Kendall Tau score out of bounds"


def run_submission_generator_demo():
    print("\n>>> [Step F] Submission Generator: End-to-End Test...")

    # Instantiate
    sub_gen = SubmissionGenerator()

    # Run generation (using cached data=False to force processing of mini-test set)
    sub_gen.generate_submission(load_cached_data=False)

    # Check output
    if os.path.exists(sub_gen.submission_path):
        df_sub = pd.read_csv(sub_gen.submission_path)
        print(f"    Submission generated at: {sub_gen.submission_path}")
        print(f"    Submission shape: {df_sub.shape}")
        assert df_sub.shape[1] == 2, "Submission should have 2 columns"
        assert "cell_order" in df_sub.columns, "Submission missing 'cell_order'"
    else:
        raise FileNotFoundError("Submission file was not created.")


if __name__ == "__main__":
    try:
        setup_demo_environment()
        create_mini_datasets()

        # Run components individually
        df_val_raw, df_val_feats, val_preds = run_pipeline_demo()

        # Verify metrics
        verify_inference_and_metric(df_val_raw, df_val_feats, val_preds)

        # Run full inference wrapper
        run_submission_generator_demo()

        print("\n>>> Demo Completed Successfully!")

    except Exception as e:
        print(f"\n!!! Demo Failed with Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

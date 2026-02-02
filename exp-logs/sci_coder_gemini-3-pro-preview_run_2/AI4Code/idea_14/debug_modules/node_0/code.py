import os
import pandas as pd
import numpy as np
import shutil
import warnings

# Import from the provided library
from library.config import Config
from library.utils import kendall_tau
from library.data_processing import NotebookLoader
from library.feature_engineering import AnchorFeatureGenerator
from library.models import Stage1Ridge, Stage2LGBM
from library.pipeline import HybridRankingPipeline


def main():
    print("=== Starting Demonstration of Notebook Cell Ordering Pipeline ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Setup
    # --------------------------------------------------------------------------
    print("\n[1] Configuring environment for rapid demonstration...")

    # Override Config paths to use a demo directory
    DEMO_DIR = "./working/demo_run"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Model Hyperparameters for Speed
    Config.VOCAB_SIZE = 1000  # Reduced from 60000
    Config.SVD_COMPONENTS = 10  # Reduced from 128
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["num_leaves"] = 8
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["verbose"] = -1

    # Set seed
    Config.set_seed(42)

    # --------------------------------------------------------------------------
    # 2. Data Subsampling (Create Mini-Datasets)
    # --------------------------------------------------------------------------
    print("\n[2] Creating mini-datasets from metadata...")

    # Load original metadata
    orig_train_meta = pd.read_csv("./metadata/train_metadata.csv")
    orig_val_meta = pd.read_csv("./metadata/val_metadata.csv")
    orig_test_meta = pd.read_csv("./metadata/test_metadata.csv")

    # Sample 25 notebooks for train, 10 for val, 10 for test
    # We select notebooks that actually exist in the input directory (sanity check handled by loader usually)
    mini_train = orig_train_meta.head(25).copy()
    mini_val = orig_val_meta.head(10).copy()
    mini_test = orig_test_meta.head(10).copy()

    # Save these mini metadata files to the demo directory
    demo_train_path = os.path.join(DEMO_DIR, "mini_train_meta.csv")
    demo_val_path = os.path.join(DEMO_DIR, "mini_val_meta.csv")
    demo_test_path = os.path.join(DEMO_DIR, "mini_test_meta.csv")

    mini_train.to_csv(demo_train_path, index=False)
    mini_val.to_csv(demo_val_path, index=False)
    mini_test.to_csv(demo_test_path, index=False)

    # Point Config to these new files
    Config.TRAIN_METADATA_PATH = demo_train_path
    Config.VAL_METADATA_PATH = demo_val_path
    Config.TEST_METADATA_PATH = demo_test_path

    print(f"    Train subset: {len(mini_train)} notebooks")
    print(f"    Val subset:   {len(mini_val)} notebooks")
    print(f"    Test subset:  {len(mini_test)} notebooks")

    # --------------------------------------------------------------------------
    # 3. Validating Data Processing (NotebookLoader)
    # --------------------------------------------------------------------------
    print("\n[3] Testing NotebookLoader...")
    loader = NotebookLoader()

    # Load flattened training data
    # force load_cached_data=False to ensure we run the processing logic
    df_train_flat = loader.get_flattened_data("train", load_cached_data=False)

    print(f"    Loaded {len(df_train_flat)} cells from training subset.")

    # Assertions
    required_cols = ["notebook_id", "cell_id", "cell_type", "source_clean", "pct_rank"]
    for col in required_cols:
        assert col in df_train_flat.columns, f"Missing column {col} in flattened data"

    # Check that we have both code and markdown
    types = df_train_flat["cell_type"].unique()
    assert (
        "code" in types and "markdown" in types
    ), "Dataset should contain both code and markdown cells"

    print("    NotebookLoader validation passed.")

    # --------------------------------------------------------------------------
    # 4. Validating Feature Engineering (AnchorFeatureGenerator)
    # --------------------------------------------------------------------------
    print("\n[4] Testing AnchorFeatureGenerator...")
    feature_gen = AnchorFeatureGenerator()

    # Generate features for the training subset
    # This also fits the internal TF-IDF and SVD models
    feats_train = feature_gen.generate_features(
        df_train_flat, "train", load_cached_data=False
    )

    print(f"    Generated features shape: {feats_train.shape}")

    # Assertions
    # The output should only contain markdown cells (targets)
    n_markdown = len(df_train_flat[df_train_flat["cell_type"] == "markdown"])
    assert (
        len(feats_train) == n_markdown
    ), f"Feature count {len(feats_train)} mismatch with markdown cells {n_markdown}"

    # Check for specific feature columns
    assert "lexical_anchor" in feats_train.columns, "Missing lexical_anchor feature"
    assert "symbolic_anchor" in feats_train.columns, "Missing symbolic_anchor feature"
    assert "svd_0" in feats_train.columns, "Missing SVD features"

    print("    AnchorFeatureGenerator validation passed.")

    # --------------------------------------------------------------------------
    # 5. Validating Models
    # --------------------------------------------------------------------------
    print("\n[5] Testing Model Classes...")

    # --- Stage 1: Ridge ---
    print("    Testing Stage1Ridge...")
    stage1 = Stage1Ridge()

    # Get TF-IDF matrix from the pipeline (already fitted in step 4)
    # We need to extract sources for markdown cells to transform
    md_mask = df_train_flat["cell_type"] == "markdown"
    md_sources = df_train_flat.loc[md_mask, "source_clean"].fillna("").tolist()

    X_tfidf, _ = feature_gen.pipeline.transform(md_sources)
    y_target = feats_train["pct_rank"].values

    # Test Fit
    stage1.fit(X_tfidf, y_target)

    # Test Predict
    preds_s1 = stage1.predict(X_tfidf)
    assert len(preds_s1) == len(y_target), "Stage 1 prediction shape mismatch"

    # Test OOF
    oof_preds = stage1.get_oof_predictions(
        X_tfidf, y_target, n_splits=3, load_cached_data=False
    )
    assert len(oof_preds) == len(y_target), "OOF prediction shape mismatch"

    print("    Stage1Ridge validation passed.")

    # --- Stage 2: LightGBM ---
    print("    Testing Stage2LGBM...")
    stage2 = Stage2LGBM()

    # Construct dummy stacked features: [Ridge_Pred, Anchor_Features...]
    # Drop metadata from feats_train
    meta_cols = ["notebook_id", "cell_id", "rank", "pct_rank"]
    feat_cols = [c for c in feats_train.columns if c not in meta_cols]
    X_anchors = feats_train[feat_cols].values

    X_stack = np.hstack([preds_s1.reshape(-1, 1), X_anchors])

    # Fit (using same data for train/val just to test API)
    stage2.fit(X_stack, y_target, X_val=X_stack, y_val=y_target)

    # Predict
    preds_s2 = stage2.predict(X_stack)
    assert len(preds_s2) == len(y_target), "Stage 2 prediction shape mismatch"

    print("    Stage2LGBM validation passed.")

    # --------------------------------------------------------------------------
    # 6. Integration Test: Full Pipeline
    # --------------------------------------------------------------------------
    print("\n[6] Running Full HybridRankingPipeline...")

    pipeline = HybridRankingPipeline()

    # 6a. Train
    # We use load_cached_data=True here to leverage the files we generated/cached in steps 3 & 4 if applicable,
    # or recompute if the cache keys differ. Given we manually called methods with False,
    # the pipeline might recompute or use what's available.
    # To be safe and test flow, we let it run.
    val_score = pipeline.train(load_cached_data=False)

    print(f"    Pipeline Training Complete. Validation Kendall Tau: {val_score:.4f}")
    assert 0.0 <= val_score <= 1.0, "Validation score out of bounds"

    # 6b. Predict
    pipeline.predict(load_cached_data=False)

    # Check submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert len(df_sub) == len(
        mini_test
    ), f"Submission row count {len(df_sub)} mismatch with test set {len(mini_test)}"
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission columns incorrect"

    print(f"    Submission generated successfully at {Config.SUBMISSION_PATH}")
    print("    First 3 rows of submission:")
    print(df_sub.head(3))

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    # Suppress specific LightGBM warnings that might clutter output
    warnings.filterwarnings("ignore", category=UserWarning)
    main()

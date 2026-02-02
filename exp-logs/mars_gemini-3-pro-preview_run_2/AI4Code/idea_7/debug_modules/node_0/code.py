import os
import sys
import shutil
import pandas as pd
import numpy as np
import warnings
import lightgbm as lgb

# Import provided library modules
from library.config import Config
from library.metrics import count_inversions, score_dataframe
from library.data_loader import load_dataset
from library.feature_engineering import FeatureEngineer
from library.modeling import StackedRanker


def main():
    print("=== Starting Demonstration Script ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Override for Speed
    # --------------------------------------------------------------------------
    print("\n[Demo] Configuring environment for fast execution...")

    # Override Config parameters to run on a tiny subset with minimal compute
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Only process 50 notebooks
    Config.VOCAB_SIZE = 500  # Small vocab for speed
    Config.SVD_COMPONENTS = 10  # Low dimensionality
    Config.N_FOLDS = 2  # Minimal CV folds
    Config.LGBM_NUM_BOOST_ROUND = 10  # Few iterations
    Config.LGBM_EARLY_STOPPING_ROUNDS = 5

    # Set demo-specific paths to avoid overwriting real work
    Config.WORKING_DIR = "./working/demo_run"
    Config.SUBMISSION_DIR = "./working/demo_submission"
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Initialize environment (creates directories, sets seeds)
    Config.setup()

    # Suppress LightGBM warnings further
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    # --------------------------------------------------------------------------
    # 2. Metrics Verification
    # --------------------------------------------------------------------------
    print("\n[Demo] Verifying Metrics Logic...")

    # Test Case 1: Perfect order
    gt = ["a", "b", "c", "d"]
    pred_perfect = ["a", "b", "c", "d"]
    inv_perfect = count_inversions(pred_perfect, gt)
    assert inv_perfect == 0, f"Expected 0 inversions, got {inv_perfect}"

    # Test Case 2: Reversed order
    pred_reversed = ["d", "c", "b", "a"]
    # Inversions: (d,c), (d,b), (d,a), (c,b), (c,a), (b,a) -> 6 inversions
    # Formula: n(n-1)/2 = 4*3/2 = 6
    inv_reversed = count_inversions(pred_reversed, gt)
    assert inv_reversed == 6, f"Expected 6 inversions, got {inv_reversed}"

    # Test Case 3: Partial swap
    pred_swap = ["a", "c", "b", "d"]  # 'c' and 'b' swapped -> 1 inversion
    inv_swap = count_inversions(pred_swap, gt)
    assert inv_swap == 1, f"Expected 1 inversion, got {inv_swap}"

    # Test Score Dataframe
    val_data = pd.DataFrame({"id": ["nb1", "nb2"], "cell_order": ["a b c", "x y"]})
    preds = {
        "nb1": ["a", "c", "b"],  # 1 swap (b,c), max swaps = 3*2 = 6. Score term: 1/6
        "nb2": ["x", "y"],  # 0 swaps, max swaps = 2*1 = 2. Score term: 0/2
    }
    # Total Swaps = 1 + 0 = 1
    # Total Max Term = 6 + 2 = 8
    # K = 1 - 4 * (1 / 8) = 1 - 0.5 = 0.5
    score = score_dataframe(val_data, preds)
    assert abs(score - 0.5) < 1e-6, f"Expected score 0.5, got {score}"

    print("Metrics verification passed.")

    # --------------------------------------------------------------------------
    # 3. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Testing Data Loader...")

    # Load training data (cached=False to force parsing)
    df_md_train, df_nb_train = load_dataset("train", load_cached_data=False)

    # Assertions
    assert not df_md_train.empty, "Markdown dataframe is empty"
    assert not df_nb_train.empty, "Notebook dataframe is empty"

    expected_md_cols = ["cell_id", "notebook_id", "source", "rank", "ancestor_id"]
    for col in expected_md_cols:
        assert col in df_md_train.columns, f"Missing column {col} in MD DataFrame"

    print(
        f"Loaded {len(df_md_train)} markdown cells from {len(df_nb_train)} notebooks."
    )

    # --------------------------------------------------------------------------
    # 4. Feature Engineering Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Testing Feature Engineering...")

    fe = FeatureEngineer(load_cached_data=False)

    # Process the loaded training split
    # This fits the pipeline and generates features
    df_final_train, _ = fe.process_split("train")

    # Assertions
    assert len(df_final_train) == len(df_md_train), "Feature DF length mismatch"

    # Check for LSA columns
    lsa_cols = [c for c in df_final_train.columns if c.startswith("lsa_")]
    assert (
        len(lsa_cols) == Config.SVD_COMPONENTS
    ), f"Expected {Config.SVD_COMPONENTS} LSA cols, found {len(lsa_cols)}"

    # Check for Anchor columns
    anchor_cols = ["anchor_rank", "anchor_sim", "top3_anchor_rank_mean"]
    for col in anchor_cols:
        assert col in df_final_train.columns, f"Missing anchor feature {col}"

    # Check for Metadata columns
    meta_cols = ["total_cells", "md_ratio"]
    for col in meta_cols:
        assert col in df_final_train.columns, f"Missing metadata feature {col}"

    print(
        f"Feature Engineering successful. Feature matrix shape: {df_final_train.shape}"
    )

    # --------------------------------------------------------------------------
    # 5. Modeling Pipeline Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Testing StackedRanker Pipeline...")

    ranker = StackedRanker()

    # We will manually trigger the stages to verify intermediate steps,
    # although ranker.run() does this automatically.

    # Load processed data (relying on cache generated in step 4 or reloading)
    df_train_feats, _ = fe.process_split("train")
    df_val_feats, _ = fe.process_split("val")

    # Ensure pipeline is loaded for transformation
    ranker.pipeline.load(Config.WORKING_DIR)

    # Stage 1: Ridge
    print("Running Stage 1 (Ridge)...")
    train_oof, val_preds, ridge_model = ranker.train_stage1(
        df_train_feats, df_val_feats, load_cached_preds=False
    )

    assert len(train_oof) == len(df_train_feats), "OOF preds length mismatch"
    assert len(val_preds) == len(df_val_feats), "Val preds length mismatch"

    # Stage 2: LightGBM
    print("Running Stage 2 (LightGBM)...")
    lgbm_model = ranker.train_stage2(df_train_feats, df_val_feats, train_oof, val_preds)

    assert lgbm_model is not None, "LightGBM model training failed"

    # Full Run (Inference)
    print("Running Inference on Test Set...")
    # We can call run() but since we already trained, let's just do the inference part manually
    # to avoid re-training, or simply call run() to demonstrate the 'one-shot' capability.
    # Let's call run() to prove the full pipeline integration works as designed.
    # Note: run() will reload models from disk if they exist, which they do now.
    ranker.run()

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    assert (
        "id" in df_sub.columns and "cell_order" in df_sub.columns
    ), "Submission format incorrect"
    assert len(df_sub) > 0, "Submission file is empty"

    print(f"Submission generated successfully at {Config.SUBMISSION_PATH}")
    print(f"Sample submission row:\n{df_sub.head(1)}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    main()

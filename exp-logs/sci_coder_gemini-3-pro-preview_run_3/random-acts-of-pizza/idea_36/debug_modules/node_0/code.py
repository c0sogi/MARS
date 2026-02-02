import sys
import os
import numpy as np
import pandas as pd
import warnings
import shutil

# Ensure library imports work
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import set_seed, Timer, get_logger
from library.feature_engineering import (
    CommunityProfiler,
    TextProcessor,
    MetadataExtractor,
)
from library.data_loader import DataLoader
from library.model_zoo import get_base_models
from library.stacking_trainer import StackingTrainer

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main():
    print(">>> Starting Demonstration Script...")

    # ==========================================
    # 1. Configuration for Speed & Reproducibility
    # ==========================================
    print("\n[1/5] Configuring Environment...")
    set_seed(42)

    # Override Config for rapid demonstration
    # We use a small sample size to ensure the pipeline runs in seconds/minutes
    Config.DEBUG_SAMPLE_SIZE = 50
    Config.N_FOLDS = 2  # Minimal folds for CV

    # Reduce model complexity for speed
    Config.MODEL_LEXICAL_RF["n_estimators"] = 5
    Config.MODEL_COMMUNITY_RF["n_estimators"] = 5
    Config.MODEL_SEMANTIC_RF["n_estimators"] = 5
    Config.MODEL_SEMANTIC_XGB["n_estimators"] = 5
    Config.MODEL_SEMANTIC_XGB["early_stopping_rounds"] = 2

    # Create necessary directories
    Config.create_dirs()
    print("Configuration updated for demo mode (Sample Size: 50, Reduced Estimators).")

    # ==========================================
    # 2. Testing Feature Engineering Components
    # ==========================================
    print("\n[2/5] Verifying Feature Engineering Logic...")

    # --- Test CommunityProfiler (Bayesian Target Encoding) ---
    print("  -> Testing CommunityProfiler...")
    profiler = CommunityProfiler(vocab_size=5, smoothing=1)

    # Synthetic data: Subreddit 'A' is 100% success, 'B' is 0% success
    # Global mean = 3/5 = 0.6
    subs_train = pd.Series([["A"], ["A"], ["B"], ["B"], ["C"]])
    y_train = np.array([1, 1, 0, 0, 1])

    profiler.fit(subs_train, y_train)

    # Transform: 'A' should be high, 'B' low, 'Z' (unseen) should be global mean
    # Calculation for A (smoothing=1): (2*1 + 1*0.6) / (2+1) = 2.6/3 ≈ 0.866
    scores = profiler.transform(pd.Series([["A"], ["B"], ["Z"]]))

    assert (
        scores[0] > scores[1]
    ), "Logic Error: High success subreddit scored lower than low success one."
    assert np.isclose(
        scores[2], 0.6
    ), "Logic Error: Unseen subreddit did not revert to global mean."
    print("     CommunityProfiler passed.")

    # --- Test TextProcessor ---
    print("  -> Testing TextProcessor...")
    df_text = pd.DataFrame({"title": ["Pizza"], "body": ["Please"]})
    processed_text = TextProcessor.process(df_text, ["title", "body"])

    assert (
        processed_text.iloc[0] == "Pizza Please"
    ), "Logic Error: Text concatenation failed."
    print("     TextProcessor passed.")

    # --- Test MetadataExtractor ---
    print("  -> Testing MetadataExtractor...")
    df_meta = pd.DataFrame({"val": [10, 20, 30, 40, 50]})
    me = MetadataExtractor()

    # Fit on data, check scaling (StandardScaler should result in mean ~0)
    transformed_meta = me.fit_transform(df_meta, ["val"])

    assert transformed_meta.shape == (5, 1), "Shape mismatch in metadata extraction."
    assert np.isclose(
        transformed_meta.mean(), 0, atol=1e-7
    ), "Scaling failed (mean != 0)."
    print("     MetadataExtractor passed.")

    # ==========================================
    # 3. Testing Data Loading Pipeline
    # ==========================================
    print("\n[3/5] Testing DataLoader...")
    loader = DataLoader()

    # Force reprocessing (load_cached_data=False) to verify engineering pipeline works
    data_dict = loader.load_dataset(
        load_cached_data=False, debug_sample_size=Config.DEBUG_SAMPLE_SIZE
    )

    # Verify dictionary structure
    required_keys = ["train", "val", "test", "CommunityProfiler"]
    for key in required_keys:
        assert key in data_dict, f"DataLoader output missing key: {key}"

    # Verify data dimensions match debug size
    train_y_len = len(data_dict["train"]["y"])
    assert (
        train_y_len <= Config.DEBUG_SAMPLE_SIZE
    ), f"Data not subsampled correctly. Got {train_y_len}"
    print(f"     DataLoader successfully processed {train_y_len} training samples.")

    # ==========================================
    # 4. Testing Model Initialization
    # ==========================================
    print("\n[4/5] Verifying Model Zoo...")
    models = get_base_models()
    expected_models = [
        "LexicalBagger",
        "CommunityBagger",
        "SemanticBooster",
        "SemanticBagger",
        "MetadataAnchor",
    ]

    for model_name in expected_models:
        assert model_name in models, f"Model Zoo missing: {model_name}"
        assert hasattr(
            models[model_name], "fit"
        ), f"{model_name} is not a valid estimator."

    print("     All base models initialized successfully.")

    # ==========================================
    # 5. Integration Test: Stacking Trainer
    # ==========================================
    print("\n[5/5] Running StackingTrainer (Full Pipeline)...")

    trainer = StackingTrainer()

    # Run the trainer. This will:
    # 1. Load the (cached) small dataset
    # 2. Perform CV Stacking
    # 3. Train Meta-Learner
    # 4. Generate Submission
    with Timer("StackingTrainer Execution"):
        trainer.run(debug_sample_size=Config.DEBUG_SAMPLE_SIZE)

    # Verify Submission
    if not os.path.exists(Config.SUBMISSION_FILE_PATH):
        raise FileNotFoundError("Submission file was not generated.")

    sub_df = pd.read_csv(Config.SUBMISSION_FILE_PATH)

    # Check submission format
    assert Config.ID_COL in sub_df.columns, "Submission missing ID column."
    assert Config.TARGET_COL in sub_df.columns, "Submission missing Target column."
    assert (
        len(sub_df) == Config.DEBUG_SAMPLE_SIZE
    ), f"Submission row count mismatch. Expected {Config.DEBUG_SAMPLE_SIZE}, got {len(sub_df)}"

    print(f"     Submission verified at: {Config.SUBMISSION_FILE_PATH}")
    print(f"     Rows: {len(sub_df)}")
    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    main()

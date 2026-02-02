import os
import pandas as pd
import numpy as np
import torch
import logging

# Import library components
from library.config import Config
from library.utils import seed_everything
from library.features import FeatureEngineer
from library.stage1_trainer import Stage1Trainer
from library.stage2_model import LGBMHandler

# Suppress verbose logs from transformers and other libs
logging.getLogger("transformers").setLevel(logging.ERROR)


def main():
    print("=== Starting Essay Scoring Pipeline Demo ===")

    # --------------------------------------------------------------------------
    # 1. Configuration & Optimization
    # --------------------------------------------------------------------------
    print("Configuring for fast execution...")

    # Enable Debug mode to use a small subset (100 samples)
    Config.DEBUG = True

    # Reduce training intensity for demonstration speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4  # Smaller batch size for the small debug set
    Config.VALID_BATCH_SIZE = 8

    # Optimize LightGBM for speed
    Config.LGBM_PARAMS["n_estimators"] = 10
    Config.LGBM_PARAMS["early_stopping_rounds"] = 5
    Config.LGBM_PARAMS["num_leaves"] = 10

    # Set global seeds
    seed_everything(Config.SEED)

    # --------------------------------------------------------------------------
    # 2. Meta-Feature Engineering
    # --------------------------------------------------------------------------
    print("\n--- Step 1: Feature Engineering ---")
    fe = FeatureEngineer()

    # Run feature extraction (Train, Val, Test)
    # This will create parquet files in working/idea_2/
    fe.run(load_cached_data=False)

    # Verification
    assert os.path.exists(Config.TRAIN_META_FEATS_PATH), "Train meta-features missing"
    assert os.path.exists(Config.TEST_META_FEATS_PATH), "Test meta-features missing"

    # Check shape of generated features
    df_meta = pd.read_parquet(Config.TRAIN_META_FEATS_PATH)
    print(f"Meta-features shape: {df_meta.shape}")
    assert (
        df_meta.shape[0] == Config.DEBUG_SAMPLES
    ), f"Expected {Config.DEBUG_SAMPLES} rows in debug mode, got {df_meta.shape[0]}"

    # --------------------------------------------------------------------------
    # 3. Stage 1: DeBERTa Fine-Tuning & Embedding Extraction
    # --------------------------------------------------------------------------
    print("\n--- Step 2: Stage 1 (DeBERTa) ---")
    trainer = Stage1Trainer()

    # Fine-tune the model
    trainer.train_deberta()

    # Verify model checkpoint
    assert os.path.exists(Config.BEST_MODEL_PATH), "Best model checkpoint missing"

    # Extract Embeddings
    # In DEBUG mode, this generates files with '_debug.npy' suffix
    embeddings = trainer.extract_embeddings(load_cached_data=False)

    # Verification
    assert embeddings["train"] is not None
    assert embeddings["test"] is not None
    print(f"Train embeddings shape: {embeddings['train'].shape}")

    # --------------------------------------------------------------------------
    # 4. Bridge: Update Paths for Stage 2
    # --------------------------------------------------------------------------
    # The Stage1Trainer automatically appends '_debug' to filenames in DEBUG mode.
    # The LGBMHandler reads directly from Config paths. We must update Config
    # to point to the debug files so LGBMHandler finds them.
    if Config.DEBUG:
        print("Updating Config paths to point to debug embeddings...")
        Config.TRAIN_EMBEDDINGS_PATH = Config.TRAIN_EMBEDDINGS_PATH.replace(
            ".npy", "_debug.npy"
        )
        Config.VAL_EMBEDDINGS_PATH = Config.VAL_EMBEDDINGS_PATH.replace(
            ".npy", "_debug.npy"
        )
        Config.TEST_EMBEDDINGS_PATH = Config.TEST_EMBEDDINGS_PATH.replace(
            ".npy", "_debug.npy"
        )

        assert os.path.exists(
            Config.TRAIN_EMBEDDINGS_PATH
        ), "Debug embedding file not found at expected path"

    # --------------------------------------------------------------------------
    # 5. Stage 2: LightGBM Training & Inference
    # --------------------------------------------------------------------------
    print("\n--- Step 3: Stage 2 (LightGBM) ---")
    lgbm_handler = LGBMHandler()

    # Train the boosting model
    lgbm_handler.train_model()

    # Generate submission
    lgbm_handler.predict_and_submit()

    # --------------------------------------------------------------------------
    # 6. Final Validation
    # --------------------------------------------------------------------------
    print("\n--- Step 4: Final Validation ---")

    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)

    # Check columns
    required_cols = {"essay_id", "score"}
    assert required_cols.issubset(
        submission_df.columns
    ), f"Missing columns. Found: {submission_df.columns}"

    # Check values
    scores = submission_df["score"]
    assert scores.min() >= 1 and scores.max() <= 6, "Scores out of range [1, 6]"
    assert pd.api.types.is_integer_dtype(scores), "Scores must be integers"

    # Check length (should match debug samples for test set)
    # Note: Test set in debug mode is also truncated to DEBUG_SAMPLES
    print(f"Submission shape: {submission_df.shape}")
    print("Sample predictions:")
    print(submission_df.head())

    print("\n=== Pipeline Execution Successful ===")


if __name__ == "__main__":
    main()

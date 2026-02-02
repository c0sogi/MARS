import os
import sys
import pandas as pd
import torch
import shutil
import logging

# Import library components
from library.config import Config
from library.utils import set_seed, setup_logger, normalize_text
from library.hfbb import HFBB
from library.data_factory import DataFactory
from library.trainer import train_transformer
from library.inference import CascadeInference


def main():
    # ==========================================
    # 1. SETUP & CONFIGURATION OVERRIDES
    # ==========================================
    print(">>> Step 1: Configuration Setup")

    # Set deterministic seed
    set_seed(42)

    # Define a working directory for this demo to avoid cluttering the main working dir
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    # Override Config paths to use the demo directory
    Config.WORKING_DIR = DEMO_DIR
    Config.HFBB_CACHE_DIR = os.path.join(DEMO_DIR, "hfbb_cache")
    Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")

    # Override Config data paths (will point to mini datasets created below)
    Config.TRAIN_META = os.path.join(DEMO_DIR, "mini_train.csv")
    Config.VAL_META = os.path.join(DEMO_DIR, "mini_val.csv")
    Config.TEST_META = os.path.join(DEMO_DIR, "mini_test.csv")

    # Override Config output paths
    Config.RESIDUAL_TRAIN_PATH = os.path.join(DEMO_DIR, "residual_train.parquet")
    Config.ENRICHED_TRAIN_PATH = os.path.join(
        DEMO_DIR, "enriched_residual_train.parquet"
    )
    Config.ENRICHED_VAL_PATH = os.path.join(DEMO_DIR, "enriched_residual_val.parquet")
    Config.BPE_MODEL_PREFIX = os.path.join(DEMO_DIR, "bpe_demo")
    Config.BEST_MODEL_PATH = os.path.join(DEMO_DIR, "transformer_best.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Override Hyperparameters for Speed
    Config.EPOCHS = 2
    Config.BATCH_SIZE = 16
    Config.N_FOLDS = 2  # Minimum folds for Jackknifing
    Config.TARGET_VOCAB_SIZE = 1000  # Smaller vocab for mini dataset
    Config.WARMUP_STEPS = 10
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for tiny data

    # Ensure directories exist
    Config.setup()

    # ==========================================
    # 2. DATA SUBSETTING (MINI DATASET)
    # ==========================================
    print("\n>>> Step 2: Creating Mini Dataset")

    # Load original metadata
    # We use the provided metadata files which are guaranteed to exist
    orig_train_path = "./metadata/train.csv"
    orig_val_path = "./metadata/val.csv"
    orig_test_path = "./metadata/test.csv"

    # Read a small chunk of data
    # We select specific sentences to ensure we have some context
    df_train_full = pd.read_csv(orig_train_path)
    df_val_full = pd.read_csv(orig_val_path)
    df_test_full = pd.read_csv(orig_test_path)

    # Select top 200 sentences for train, 50 for val, 50 for test
    train_sents = df_train_full["sentence_id"].unique()[:200]
    val_sents = df_val_full["sentence_id"].unique()[:50]
    test_sents = df_test_full["sentence_id"].unique()[:50]

    df_mini_train = df_train_full[df_train_full["sentence_id"].isin(train_sents)].copy()
    df_mini_val = df_val_full[df_val_full["sentence_id"].isin(val_sents)].copy()
    df_mini_test = df_test_full[df_test_full["sentence_id"].isin(test_sents)].copy()

    # Save mini datasets
    df_mini_train.to_csv(Config.TRAIN_META, index=False)
    df_mini_val.to_csv(Config.VAL_META, index=False)
    df_mini_test.to_csv(Config.TEST_META, index=False)

    print(f"Mini Train Shape: {df_mini_train.shape}")
    print(f"Mini Val Shape: {df_mini_val.shape}")
    print(f"Mini Test Shape: {df_mini_test.shape}")

    # ==========================================
    # 3. HFBB (TIER 1) DEMONSTRATION
    # ==========================================
    print("\n>>> Step 3: HFBB (Hierarchical Frequency Back-off) Demo")

    # Instantiate HFBB
    hfbb = HFBB()

    # Fit on mini training data
    # load_cached_data=False forces it to build maps from our new mini dataset
    hfbb.fit(df_mini_train, load_cached_data=False)

    # Verify maps are populated
    assert hfbb.trigram_map is not None
    assert hfbb.unigram_map is not None
    print(f"HFBB Trigram Map Size: {len(hfbb.trigram_map)}")
    print(f"HFBB Unigram Map Size: {len(hfbb.unigram_map)}")

    # Test Prediction on a small batch
    sample_batch = df_mini_val.head(10).copy()
    preds = hfbb.predict_batch(sample_batch)

    print("HFBB Sample Predictions:")
    print(
        pd.concat(
            [sample_batch[["before", "after"]], preds.rename("prediction")], axis=1
        )
    )

    # ==========================================
    # 4. DATA FACTORY & CURRICULUM GENERATION
    # ==========================================
    print("\n>>> Step 4: Data Factory & Curriculum Generation")

    factory = DataFactory()

    # Train BPE Tokenizer
    factory.train_bpe_tokenizer(df_mini_train)
    assert os.path.exists(Config.BPE_MODEL_PREFIX + ".model"), "BPE Model not created"

    # Generate Curriculum Data (Residuals + Anchors)
    # This runs Jackknifing (N_FOLDS=2) on the mini train set
    df_enriched = factory.generate_curriculum_data(
        df_mini_train, load_cached_data=False
    )

    print(f"Enriched Training Data Size: {len(df_enriched)}")
    print("Columns:", df_enriched.columns.tolist())

    # Verify we have residuals or anchors
    # Note: With such small data, we might not find complex residuals, but the function should return a dataframe
    assert not df_enriched.empty, "Enriched dataset is empty!"

    # Prepare Validation Data (Filtered for Transformer relevance)
    df_val_filtered = factory.prepare_val_data(
        df_mini_train, df_mini_val, load_cached_data=False
    )
    print(f"Filtered Validation Data Size: {len(df_val_filtered)}")

    # ==========================================
    # 5. TRANSFORMER (TIER 2) TRAINING
    # ==========================================
    print("\n>>> Step 5: Transformer Training")

    # We use the high-level train_transformer function which orchestrates everything.
    # It will reload the data we just prepared (since we saved it to the Config paths)
    # or regenerate it if cache was not hit (but we just generated cache).
    # Since we set load_cached_data=True in the function call below, it should pick up the files we just made.

    train_transformer(load_cached_data=True)

    assert os.path.exists(Config.BEST_MODEL_PATH), "Model checkpoint was not saved!"
    print(f"Model saved to {Config.BEST_MODEL_PATH}")

    # ==========================================
    # 6. INFERENCE PIPELINE & SUBMISSION
    # ==========================================
    print("\n>>> Step 6: Inference Pipeline")

    # Instantiate Inference Engine
    # This loads HFBB (from cache) and Transformer (from checkpoint)
    inference_engine = CascadeInference()

    # Run prediction on the mini test set
    # The generate_submission method reads from Config.TEST_META
    inference_engine.generate_submission()

    # Verify Submission
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission Shape: {submission_df.shape}")
    print("Submission Head:")
    print(submission_df.head())

    # Assertions
    assert len(submission_df) == len(df_mini_test), "Submission row count mismatch"
    assert "id" in submission_df.columns and "after" in submission_df.columns

    # Check specific ID format
    first_id = submission_df.iloc[0]["id"]
    assert "_" in str(first_id), f"Invalid ID format: {first_id}"

    print("\n>>> Demo Completed Successfully!")


if __name__ == "__main__":
    main()

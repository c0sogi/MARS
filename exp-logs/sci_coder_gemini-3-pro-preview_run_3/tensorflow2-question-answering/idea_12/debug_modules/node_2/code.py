import os
import pandas as pd
import torch
import numpy as np
import shutil

# Import library components
from library.config import Config
from library.text_utils import build_or_load_vocab, Vocab
from library.trainer import train_ranker, train_reader
from library.evaluator import predict_submission
from library.data_loader import get_ranker_loaders, get_reader_loaders
from library.ranker_net import prepare_ranker_data
from library.reader_net import prepare_reader_data


def run_demo():
    print("=== Starting Demonstration of NQ Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configure for Speed (Demo Mode)
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for demo execution...")

    # Override Config parameters to use a temporary working directory and small data subsets
    Config.WORKING_DIR = "./working/demo_execution"
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")

    # Update derived paths in Config based on new WORKING_DIR
    Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.parquet")
    Config.EMBEDDING_MATRIX_PATH = os.path.join(
        Config.WORKING_DIR, "embedding_matrix.npy"
    )
    Config.RANKER_TRAIN_PATH = os.path.join(
        Config.WORKING_DIR, "ranker_train_data.parquet"
    )
    Config.RANKER_VAL_PATH = os.path.join(Config.WORKING_DIR, "ranker_val_data.parquet")
    Config.RANKER_TEST_PATH = os.path.join(
        Config.WORKING_DIR, "ranker_test_features.parquet"
    )
    Config.READER_TRAIN_PATH = os.path.join(
        Config.WORKING_DIR, "reader_train_data.parquet"
    )
    Config.READER_VAL_PATH = os.path.join(Config.WORKING_DIR, "reader_val_data.parquet")
    Config.READER_TEST_PATH = os.path.join(
        Config.WORKING_DIR, "reader_test_features.parquet"
    )
    Config.RANKER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "reader_best.pth")
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Reduce hyperparameters for speed
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 4
    Config.TRAIN_SAMPLE_SIZE = 50  # Use only 50 samples for training
    Config.VAL_SAMPLE_SIZE = 10  # Use only 10 samples for validation
    Config.VOCAB_SIZE = 1000  # Small vocab for demo
    Config.EMBEDDING_DIM = 16  # Small embedding dim

    # Ensure directories exist
    Config.setup_directories()

    print(f"Working directory set to: {Config.WORKING_DIR}")

    # -------------------------------------------------------------------------
    # 2. Vocabulary Generation
    # -------------------------------------------------------------------------
    print("\n[2] Building Vocabulary...")

    # Load metadata to pass to vocab builder
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)

    # Force rebuild of vocab for the demo to ensure it matches our small embedding dim
    if os.path.exists(Config.VOCAB_PATH):
        os.remove(Config.VOCAB_PATH)
    if os.path.exists(Config.EMBEDDING_MATRIX_PATH):
        os.remove(Config.EMBEDDING_MATRIX_PATH)

    vocab = build_or_load_vocab(train_meta, load_cached_data=False)

    # Verify Vocab
    assert isinstance(vocab, Vocab), "Returned object is not a Vocab instance"
    assert os.path.exists(Config.VOCAB_PATH), "Vocab file not saved"
    assert os.path.exists(Config.EMBEDDING_MATRIX_PATH), "Embedding matrix not saved"
    assert vocab.embedding_matrix.shape == (
        vocab.vocab_size,
        Config.EMBEDDING_DIM,
    ), f"Embedding shape mismatch: {vocab.embedding_matrix.shape}"
    print("Vocabulary built and verified.")

    # -------------------------------------------------------------------------
    # 3. Ranker Model Training
    # -------------------------------------------------------------------------
    print("\n[3] Training Ranker Model...")

    # Verify data preparation logic explicitly before training
    print("Verifying Ranker data preparation logic...")
    train_meta_sample = train_meta.sample(n=10, random_state=Config.SEED)
    ranker_df_sample = prepare_ranker_data(
        train_meta_sample,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=False,
    )
    assert (
        not ranker_df_sample.empty
    ), "Ranker data preparation returned empty DataFrame"
    assert (
        "q_ids" in ranker_df_sample.columns and "p_ids" in ranker_df_sample.columns
    ), "Ranker DataFrame missing required columns"
    assert "label" in ranker_df_sample.columns, "Ranker training data missing labels"

    # Run training (this handles data loading internally via get_ranker_loaders)
    ranker_model = train_ranker(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        patience=1,
        train_sample_size=Config.TRAIN_SAMPLE_SIZE,
        val_sample_size=Config.VAL_SAMPLE_SIZE,
    )

    assert os.path.exists(
        Config.RANKER_MODEL_PATH
    ), "Ranker model checkpoint not found after training"
    print("Ranker training completed and model saved.")

    # -------------------------------------------------------------------------
    # 4. Reader Model Training
    # -------------------------------------------------------------------------
    print("\n[4] Training Reader Model...")

    # Verify Reader data preparation logic
    print("Verifying Reader data preparation logic...")
    # Note: Reader requires samples that actually have short answers.
    # Sampling randomly might miss them if they are rare, but stratified split helps.
    # We use the library function which handles filtering.
    reader_df_sample = prepare_reader_data(
        train_meta_sample,
        vocab,
        Config.TRAIN_RAW_FILE,
        is_train=True,
        load_cached_data=False,
    )
    # It's possible for this small sample to be empty if no short answers exist in the 10 rows.
    # If empty, we skip assertion on content, but check type.
    assert isinstance(reader_df_sample, pd.DataFrame)

    # Run training
    reader_model = train_reader(
        epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-3,
        patience=1,
        train_sample_size=Config.TRAIN_SAMPLE_SIZE,
        val_sample_size=Config.VAL_SAMPLE_SIZE,
    )

    assert os.path.exists(
        Config.READER_MODEL_PATH
    ), "Reader model checkpoint not found after training"
    print("Reader training completed and model saved.")

    # -------------------------------------------------------------------------
    # 5. Full Inference Pipeline
    # -------------------------------------------------------------------------
    print("\n[5] Running Inference Pipeline (Ranker + Reader)...")

    # To speed up test inference, we will create a small dummy test metadata file
    # pointing to the real test file but only containing a few examples.
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    demo_test_meta = full_test_meta.head(20)  # Take first 20 examples
    demo_test_meta_path = os.path.join(Config.WORKING_DIR, "demo_test_metadata.csv")
    demo_test_meta.to_csv(demo_test_meta_path, index=False)

    # Temporarily point Config to this small test metadata
    original_test_meta_path = Config.TEST_METADATA_PATH
    Config.TEST_METADATA_PATH = demo_test_meta_path

    try:
        # Run the full prediction pipeline
        # We pass load_cached_data=False to force it to run using our new models
        predict_submission(load_cached_data=False)

        # Verify Submission
        assert os.path.exists(
            Config.SUBMISSION_FILE
        ), "Submission file was not generated"

        sub_df = pd.read_csv(Config.SUBMISSION_FILE)
        print(f"Submission generated with {len(sub_df)} rows.")

        # Check format
        assert "example_id" in sub_df.columns, "Submission missing example_id"
        assert (
            "PredictionString" in sub_df.columns
        ), "Submission missing PredictionString"

        # We expect 2 rows per example (long and short)
        expected_rows = len(demo_test_meta) * 2
        assert (
            len(sub_df) == expected_rows
        ), f"Expected {expected_rows} rows in submission, got {len(sub_df)}"

        print("Submission format verified.")

    finally:
        # Restore config path
        Config.TEST_METADATA_PATH = original_test_meta_path

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

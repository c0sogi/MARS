import sys
import os
import shutil
import pandas as pd
import torch
import numpy as np

# Add current directory to sys.path to ensure library imports work
sys.path.append(os.getcwd())

# Import Config and modify for demonstration speed
from library.config import Config

# Modify Config for fast execution
Config.WORKING_DIR = "./working/demo_execution"
Config.SAMPLE_SIZE = 100  # Small sample for speed
Config.VOCAB_SIZE = 1000
Config.EMBEDDING_DIM = 32  # Smaller embedding for speed
Config.BATCH_SIZE = 8
Config.EPOCHS = 1
Config.RANKER_FILTERS = 16
Config.RANKER_HIDDEN_DIM = 32
Config.READER_FILTERS = 16
Config.EARLY_STOPPING_PATIENCE = 1

# Update paths based on new working dir
Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.parquet")
Config.EMBEDDING_MATRIX_PATH = os.path.join(Config.WORKING_DIR, "embedding_matrix.npy")
Config.RANKER_TRAIN_DATA_PATH = os.path.join(
    Config.WORKING_DIR, "ranker_train_data.parquet"
)
Config.RANKER_VAL_DATA_PATH = os.path.join(
    Config.WORKING_DIR, "ranker_val_data.parquet"
)
Config.READER_TRAIN_DATA_PATH = os.path.join(
    Config.WORKING_DIR, "reader_train_data.parquet"
)
Config.READER_VAL_DATA_PATH = os.path.join(
    Config.WORKING_DIR, "reader_val_data.parquet"
)
Config.RANKER_TEST_FEATURES_PATH = os.path.join(
    Config.WORKING_DIR, "ranker_test_features.parquet"
)
Config.RANKER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ranker_best.pth")
Config.READER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "reader_best.pth")
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Ensure directories exist
Config.ensure_directories()

# Import other modules after config modification
from library.data_utils import prepare_data, get_ranker_datasets, get_reader_datasets
from library.models import EarlyFusionRanker, DynamicKernelReader
from library.trainers import Trainer
from library.inference import InferencePipeline


def demonstrate_data_processing():
    print("\n=== Demonstrating Data Processing ===")

    # Run data preparation
    # load_cached_data=False forces regeneration using the small SAMPLE_SIZE
    prepare_data(load_cached_data=False)

    # Validation
    assert os.path.exists(Config.VOCAB_PATH), "Vocab file not created"
    assert os.path.exists(Config.EMBEDDING_MATRIX_PATH), "Embeddings not created"
    assert os.path.exists(
        Config.RANKER_TRAIN_DATA_PATH
    ), "Ranker train data not created"

    # Check dataset loading
    ranker_train, ranker_val = get_ranker_datasets(load_cached_data=True)
    # Reader data might be empty if the small sample contains no short answers, handle gracefully
    try:
        reader_train, reader_val = get_reader_datasets(load_cached_data=True)
        print(f"Reader Train Size: {len(reader_train)}")
    except Exception:
        print(
            "Reader dataset generation yielded no samples (expected with very small random subset)."
        )
        reader_train = []

    print(f"Ranker Train Size: {len(ranker_train)}")

    # Check item structure if data exists
    if len(ranker_train) > 0:
        item = ranker_train[0]
        assert "input_ids" in item
        assert "label" in item
        assert item["input_ids"].shape[0] == Config.MAX_RANKER_SEQ_LEN

    if len(reader_train) > 0:
        item = reader_train[0]
        assert "q_input_ids" in item
        assert "ctx_input_ids" in item
        assert "start_idx" in item

    print("Data processing validation successful.")


def demonstrate_models():
    print("\n=== Demonstrating Models (Forward Pass) ===")

    # Load embedding matrix created during data processing
    if os.path.exists(Config.EMBEDDING_MATRIX_PATH):
        embedding_matrix = np.load(Config.EMBEDDING_MATRIX_PATH)
    else:
        # Fallback if data processing didn't run fully
        embedding_matrix = np.random.rand(Config.VOCAB_SIZE, Config.EMBEDDING_DIM)

    # 1. EarlyFusionRanker
    ranker = EarlyFusionRanker(embedding_matrix=embedding_matrix)
    # Create dummy input: (Batch=2, Seq_Len)
    dummy_input = torch.randint(0, Config.VOCAB_SIZE, (2, Config.MAX_RANKER_SEQ_LEN))

    ranker.eval()
    with torch.no_grad():
        logits = ranker(dummy_input)

    assert logits.shape == (2,), f"Ranker output shape mismatch: {logits.shape}"
    print("Ranker forward pass successful.")

    # 2. DynamicKernelReader
    reader = DynamicKernelReader(embedding_matrix=embedding_matrix)
    # Create dummy inputs
    q_input = torch.randint(0, Config.VOCAB_SIZE, (2, Config.MAX_QUESTION_LEN))
    ctx_input = torch.randint(0, Config.VOCAB_SIZE, (2, Config.MAX_PARAGRAPH_LEN))

    reader.eval()
    with torch.no_grad():
        start_logits, end_logits = reader(q_input, ctx_input)

    assert start_logits.shape == (
        2,
        Config.MAX_PARAGRAPH_LEN,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        2,
        Config.MAX_PARAGRAPH_LEN,
    ), f"End logits shape mismatch: {end_logits.shape}"
    print("Reader forward pass successful.")


def demonstrate_training():
    print("\n=== Demonstrating Training Loop ===")

    trainer = Trainer()

    # Train Ranker
    # We use load_cached_data=True because we generated data in previous step
    trainer.train_ranker(load_cached_data=True)
    assert os.path.exists(Config.RANKER_MODEL_PATH), "Ranker model checkpoint not saved"

    # Train Reader
    # Only train if we have data
    if os.path.exists(Config.READER_TRAIN_DATA_PATH):
        df = pd.read_parquet(Config.READER_TRAIN_DATA_PATH)
        if len(df) > 0:
            trainer.train_reader(load_cached_data=True)
            assert os.path.exists(
                Config.READER_MODEL_PATH
            ), "Reader model checkpoint not saved"
        else:
            print("Skipping Reader training due to empty dataset.")
            # Create dummy model file for inference step
            model = DynamicKernelReader()
            torch.save(model.state_dict(), Config.READER_MODEL_PATH)

    print("Training demonstration successful.")


def demonstrate_inference():
    print("\n=== Demonstrating Inference Pipeline ===")

    # Create a small subset of test metadata for quick inference
    original_test_meta_path = "./metadata/test_metadata.csv"
    if os.path.exists(original_test_meta_path):
        df_test = pd.read_csv(original_test_meta_path)
        # Take top 20 examples
        df_demo_test = df_test.head(20)

        demo_test_meta_path = os.path.join(Config.WORKING_DIR, "demo_test_metadata.csv")
        df_demo_test.to_csv(demo_test_meta_path, index=False)

        # Point Config to this new file
        Config.TEST_METADATA_PATH = demo_test_meta_path
    else:
        print("Original test metadata not found, skipping inference run on file.")
        return

    # Initialize Pipeline
    pipeline = InferencePipeline(load_cached_data=True)

    # Test predict_single with dummy data
    dummy_data = {
        "document_text": "This is a sample document text with <P> a paragraph </P> containing the answer.",
        "question_text": "sample document",
    }
    long_pred, short_pred = pipeline.predict_single(dummy_data)
    print(f"Single Prediction - Long: '{long_pred}', Short: '{short_pred}'")

    # Run full inference on demo test set
    pipeline.run_inference(load_cached_data=True)

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify submission format
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "example_id" in sub_df.columns
    assert "PredictionString" in sub_df.columns
    assert len(sub_df) == 40, f"Expected 40 rows (20 examples * 2), got {len(sub_df)}"

    print("Inference demonstration successful.")


if __name__ == "__main__":
    # Clean up working directory if it exists to ensure fresh run
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)

    try:
        demonstrate_data_processing()
        demonstrate_models()
        demonstrate_training()
        demonstrate_inference()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        raise e

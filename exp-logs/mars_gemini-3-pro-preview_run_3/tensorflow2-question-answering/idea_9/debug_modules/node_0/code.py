import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
from library.preprocessing import TextPreprocessor
from library.data_loader import NQRankerDataset, NQReaderDataset, collate_fn
from library.ranker_model import DecomposableAttentionRanker
from library.reader_model import GatedConvReader
from library.train_engine import train_ranker, train_reader
from library.eval_engine import EvalEngine


def setup_demo_config():
    """
    Modifies the global Config to use a temporary working directory
    and small sample sizes for rapid execution.
    """
    print("Setting up demonstration configuration...")

    # Define a specific working directory for this demo
    Config.WORKING_DIR = "./working/demo_execution"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths dependent on WORKING_DIR
    Config.VOCAB_CACHE = os.path.join(Config.WORKING_DIR, "vocab.parquet")
    Config.EMBEDDING_MATRIX_CACHE = os.path.join(
        Config.WORKING_DIR, "embedding_matrix.npy"
    )
    Config.RANKER_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "ranker_train_data.parquet"
    )
    Config.RANKER_VAL_CACHE = os.path.join(
        Config.WORKING_DIR, "ranker_val_data.parquet"
    )
    Config.RANKER_TEST_FEATURES_CACHE = os.path.join(
        Config.WORKING_DIR, "ranker_test_features.parquet"
    )
    Config.READER_TRAIN_CACHE = os.path.join(
        Config.WORKING_DIR, "reader_train_data.parquet"
    )
    Config.READER_VAL_CACHE = os.path.join(
        Config.WORKING_DIR, "reader_val_data.parquet"
    )
    Config.RANKER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "reader_best.pth")

    # Update Submission Directory
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
    Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Hyperparameters for speed
    Config.SAMPLE_SIZE = 100  # Only use 100 samples for training/validation
    Config.NUM_EPOCHS = 1  # Only run 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VOCAB_SIZE = 1000  # Smaller vocab for demo
    Config.EMBED_DIM = 16  # Smaller embedding dim for demo speed
    Config.RANKER_HIDDEN_SIZE = 32
    Config.READER_FILTERS = 16

    # Create a small subset of test metadata to speed up inference demo
    full_test_meta = pd.read_csv(Config.TEST_METADATA)
    small_test_meta = full_test_meta.head(20)  # Only 20 test samples
    demo_test_meta_path = os.path.join(Config.WORKING_DIR, "demo_test_metadata.csv")
    small_test_meta.to_csv(demo_test_meta_path, index=False)
    Config.TEST_METADATA = demo_test_meta_path

    print(f"Config updated. Working dir: {Config.WORKING_DIR}")


def demonstrate_preprocessing():
    print("\n--- Demonstrating Preprocessing ---")
    preprocessor = TextPreprocessor()

    # 1. Build Vocabulary
    # Force rebuild (load_cached_data=False) to use our small SAMPLE_SIZE
    vocab = preprocessor.build_vocabulary(load_cached_data=False)
    print(f"Vocabulary built. Size: {len(vocab)}")

    # Validation
    assert Config.PAD_TOKEN in vocab
    assert Config.UNK_TOKEN in vocab
    assert len(vocab) <= Config.VOCAB_SIZE

    # 2. Load Embeddings
    # Force rebuild to match new vocab and dim
    embedding_matrix = preprocessor.load_embeddings(load_cached_data=False)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    # Validation
    assert embedding_matrix.shape == (len(vocab), Config.EMBED_DIM)

    # 3. Text to Indices
    sample_text = "who is the president"
    indices = preprocessor.text_to_indices(sample_text, max_len=10)
    print(f"Text: '{sample_text}' -> Indices: {indices}")

    assert indices.shape == (10,)
    assert isinstance(indices, torch.LongTensor)

    return preprocessor, embedding_matrix


def demonstrate_ranker_training(preprocessor, embedding_matrix):
    print("\n--- Demonstrating Ranker Training ---")

    # 1. Create Datasets
    # We use load_cached_data=False to ensure we process the small sample defined in Config
    train_dataset = NQRankerDataset(
        metadata_path=Config.TRAIN_METADATA,
        raw_file=Config.TRAIN_FILE,
        preprocessor=preprocessor,
        is_train=True,
        load_cached_data=False,
        sample_size=Config.SAMPLE_SIZE,
    )

    val_dataset = NQRankerDataset(
        metadata_path=Config.VAL_METADATA,
        raw_file=Config.TRAIN_FILE,
        preprocessor=preprocessor,
        is_train=False,
        load_cached_data=False,
        sample_size=Config.SAMPLE_SIZE,
    )

    print(f"Ranker Train Dataset size: {len(train_dataset)}")
    print(f"Ranker Val Dataset size: {len(val_dataset)}")

    if len(train_dataset) == 0:
        print(
            "Warning: Ranker train dataset is empty (likely due to sampling filtering). Skipping training."
        )
        return

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 3. Verify Batch
    batch = next(iter(train_loader))
    print(f"Ranker Batch Keys: {batch.keys()}")
    print(f"Q Indices Shape: {batch['q_indices'].shape}")
    print(f"Doc Indices Shape: {batch['doc_indices'].shape}")
    print(f"Labels Shape: {batch['labels'].shape}")

    assert "q_indices" in batch
    assert "doc_indices" in batch
    assert "labels" in batch

    # 4. Train Model
    model = train_ranker(train_loader, val_loader, embedding_matrix, Config.DEVICE)

    # Verify model artifact
    assert os.path.exists(Config.RANKER_MODEL_PATH)
    print("Ranker model training complete and checkpoint saved.")


def demonstrate_reader_training(preprocessor, embedding_matrix):
    print("\n--- Demonstrating Reader Training ---")

    # 1. Create Datasets
    train_dataset = NQReaderDataset(
        metadata_path=Config.TRAIN_METADATA,
        raw_file=Config.TRAIN_FILE,
        preprocessor=preprocessor,
        is_train=True,
        load_cached_data=False,
        sample_size=Config.SAMPLE_SIZE,
    )

    val_dataset = NQReaderDataset(
        metadata_path=Config.VAL_METADATA,
        raw_file=Config.TRAIN_FILE,
        preprocessor=preprocessor,
        is_train=False,
        load_cached_data=False,
        sample_size=Config.SAMPLE_SIZE,
    )

    print(f"Reader Train Dataset size: {len(train_dataset)}")

    if len(train_dataset) == 0:
        print("Warning: Reader train dataset is empty. Skipping training.")
        return

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    # 3. Verify Batch
    batch = next(iter(train_loader))
    print(f"Reader Batch Keys: {batch.keys()}")
    print(f"Input Indices Shape: {batch['input_indices'].shape}")
    print(f"Start Targets Shape: {batch['start_positions'].shape}")

    assert "input_indices" in batch
    assert "start_positions" in batch
    assert "end_positions" in batch

    # 4. Train Model
    model = train_reader(train_loader, val_loader, embedding_matrix, Config.DEVICE)

    # Verify model artifact
    assert os.path.exists(Config.READER_MODEL_PATH)
    print("Reader model training complete and checkpoint saved.")


def demonstrate_evaluation():
    print("\n--- Demonstrating Evaluation Pipeline ---")

    # The EvalEngine initializes models and loads checkpoints automatically.
    # It expects the checkpoints to exist (which we created in previous steps).

    eval_engine = EvalEngine()

    # Run prediction on the small test metadata we created
    # load_cached_data=False to force processing of our new demo test metadata
    eval_engine.predict_sample(load_cached_data=False)

    # Verify Submission
    assert os.path.exists(Config.SUBMISSION_FILE)
    df = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"Submission file created with {len(df)} rows.")
    print("Head of submission:")
    print(df.head())

    # Basic format check
    assert "example_id" in df.columns
    assert "PredictionString" in df.columns


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Preprocessing
    preprocessor, embedding_matrix = demonstrate_preprocessing()

    # 3. Ranker
    demonstrate_ranker_training(preprocessor, embedding_matrix)

    # 4. Reader
    demonstrate_reader_training(preprocessor, embedding_matrix)

    # 5. Evaluation
    demonstrate_evaluation()

    print("\nAll demonstrations completed successfully.")

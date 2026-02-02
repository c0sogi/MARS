import os
import sys
import torch
import pandas as pd
import numpy as np
import random
from torch.utils.data import DataLoader

# Ensure the current directory is in the python path to import library modules
sys.path.append(os.getcwd())

# Import from provided library files
from library.utils import (
    Tokenizer,
    load_embeddings,
    CACHE_DIR,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MAX_SEQ_LEN,
    get_dataset_partitions,
)
from library.models import CompareAggregateRanker, DilatedConvReader
from library.data_loader import (
    get_tokenizer,
    process_ranker_data,
    process_reader_data,
    NQRankerDataset,
    NQReaderDataset,
    ranker_collate_fn,
    reader_collate_fn,
)
from library.trainer import ModelTrainer
from library.inference import InferencePipeline

# Set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


def demo_utils_and_models():
    print("\n=== 1. Demonstrating Utils and Models ===")

    # --- Tokenizer Demo ---
    texts = ["hello world", "hello universe", "world of ai"]
    tokenizer = Tokenizer(vocab_size=100)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts, max_len=5)

    print(f"Tokenizer vocab size: {tokenizer.vocab_count}")
    assert tokenizer.vocab_count > 2, "Vocabulary should contain more than PAD and UNK"
    assert len(sequences) == 3, "Should have 3 sequences"
    assert len(sequences[0]) == 5, "Sequence length should be padded to max_len"

    # --- Embeddings Demo ---
    # Create a dummy embedding matrix
    embedding_matrix = load_embeddings(
        tokenizer.word_index,
        embedding_dim=16,
        load_cached_data=False,  # Force recompute for demo
    )
    assert embedding_matrix.shape == (tokenizer.vocab_count, 16)

    # --- Ranker Model Demo ---
    ranker = CompareAggregateRanker(embedding_matrix, hidden_dim=32)
    # Batch size 2, seq len 5
    q_ids = torch.tensor([[1, 2, 0, 0, 0], [2, 3, 4, 0, 0]], dtype=torch.long)
    p_ids = torch.tensor([[2, 3, 4, 1, 0], [1, 2, 3, 0, 0]], dtype=torch.long)

    ranker_out = ranker(q_ids, p_ids)
    print(f"Ranker output shape: {ranker_out.shape}")
    assert ranker_out.shape == (2, 1), "Ranker output should be (batch_size, 1)"

    # --- Reader Model Demo ---
    reader = DilatedConvReader(embedding_matrix, hidden_dim=32, num_layers=2)
    # Batch size 2, seq len 10
    input_ids = torch.randint(0, tokenizer.vocab_count, (2, 10))
    start_logits, end_logits = reader(input_ids)

    print(f"Reader output shapes: {start_logits.shape}, {end_logits.shape}")
    assert start_logits.shape == (2, 10), "Start logits shape mismatch"
    assert end_logits.shape == (2, 10), "End logits shape mismatch"


def demo_data_loading_and_training():
    print("\n=== 2. Demonstrating Data Loading and Training ===")

    # Load metadata (assuming metadata generation script has run)
    train_meta, val_meta, _ = get_dataset_partitions()

    # Use a tiny sample for speed
    SAMPLE_SIZE = 50
    BATCH_SIZE = 4

    print(f"Using sample size: {SAMPLE_SIZE}")

    # Build Tokenizer from real data
    # Force reload/rebuild to ensure it works with the sample
    tokenizer = get_tokenizer(
        train_meta, load_cached_data=False, sample_size=SAMPLE_SIZE
    )

    # Load Embeddings
    embedding_matrix = load_embeddings(tokenizer.word_index, load_cached_data=False)

    # --- Process Ranker Data ---
    ranker_train_df = process_ranker_data(
        train_meta,
        tokenizer,
        load_cached_data=False,
        split="demo_train",
        sample_size=SAMPLE_SIZE,
    )
    ranker_val_df = process_ranker_data(
        val_meta,
        tokenizer,
        load_cached_data=False,
        split="demo_val",
        sample_size=SAMPLE_SIZE,
    )

    assert not ranker_train_df.empty, "Ranker training dataframe is empty"

    # Create Datasets and Loaders
    ranker_train_ds = NQRankerDataset(ranker_train_df)
    ranker_val_ds = NQRankerDataset(ranker_val_df)

    ranker_train_loader = DataLoader(
        ranker_train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=ranker_collate_fn,
    )
    ranker_val_loader = DataLoader(
        ranker_val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=ranker_collate_fn,
    )

    # Initialize Trainer and Ranker Model
    trainer = ModelTrainer()
    ranker_model = CompareAggregateRanker(embedding_matrix)

    # Train Ranker (1 epoch)
    print("Training Ranker...")
    trainer.train_ranker(ranker_model, ranker_train_loader, ranker_val_loader, epochs=1)

    # Verify model saved
    assert os.path.exists(
        os.path.join(CACHE_DIR, "ranker_best.pth")
    ), "Ranker model checkpoint not found"

    # --- Process Reader Data ---
    reader_train_df = process_reader_data(
        train_meta,
        tokenizer,
        load_cached_data=False,
        split="demo_train",
        sample_size=SAMPLE_SIZE,
    )
    reader_val_df = process_reader_data(
        val_meta,
        tokenizer,
        load_cached_data=False,
        split="demo_val",
        sample_size=SAMPLE_SIZE,
    )

    # Create Datasets and Loaders
    if not reader_train_df.empty:
        reader_train_ds = NQReaderDataset(reader_train_df)
        reader_val_ds = NQReaderDataset(reader_val_df)

        reader_train_loader = DataLoader(
            reader_train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=reader_collate_fn,
        )
        reader_val_loader = DataLoader(
            reader_val_ds,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=reader_collate_fn,
        )

        # Initialize Reader Model
        reader_model = DilatedConvReader(embedding_matrix)

        # Train Reader (1 epoch)
        print("Training Reader...")
        trainer.train_reader(
            reader_model, reader_train_loader, reader_val_loader, epochs=1
        )

        # Verify model saved
        assert os.path.exists(
            os.path.join(CACHE_DIR, "reader_best.pth")
        ), "Reader model checkpoint not found"
    else:
        print("Warning: Not enough short answers in sample to train reader.")


def demo_inference():
    print("\n=== 3. Demonstrating Inference Pipeline ===")

    SAMPLE_SIZE = 10

    # Initialize Pipeline
    pipeline = InferencePipeline()

    # Load Resources (will use the models trained in step 2)
    # We set load_cached_data=True to pick up the vocab/embeddings we just created
    pipeline.load_resources(load_cached_data=True)

    # Preprocess Test Data
    # We force re-processing to ensure we use the sample size
    test_data = pipeline.preprocess_test_data(
        load_cached_data=False, sample_size=SAMPLE_SIZE
    )

    assert len(test_data) > 0, "No test data processed"
    print(f"Processed {len(test_data)} test examples.")

    # Run Prediction
    predictions = pipeline.predict(test_data)

    assert (
        len(predictions) == len(test_data) * 2
    ), "Should have 2 predictions (long/short) per example"
    print(f"Generated {len(predictions)} prediction rows.")

    # Generate Submission (End-to-End wrapper)
    # Note: This will overwrite the predictions variable above, but demonstrates the full method
    pipeline.generate_submission(sample_size=SAMPLE_SIZE, load_cached_data=True)

    submission_path = os.path.join(pipeline.submission_dir, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not created"

    # Verify submission content
    df = pd.read_csv(submission_path)
    print("Submission Head:")
    print(df.head())
    assert "example_id" in df.columns and "PredictionString" in df.columns


if __name__ == "__main__":
    # Ensure metadata exists before running
    if not os.path.exists("./metadata/train_metadata.csv"):
        print(
            "Error: Metadata not found. Please run the metadata generation script first."
        )
        sys.exit(1)

    try:
        demo_utils_and_models()
        demo_data_loading_and_training()
        demo_inference()
        print("\nAll demonstrations completed successfully.")
    except Exception as e:
        print(f"\nAn error occurred during execution: {e}")
        raise e

import os
import sys
import torch
import pandas as pd
import numpy as np
import shutil

# --------------------------------------------------------------------------
# 1. Configuration Setup
# --------------------------------------------------------------------------
# We modify the Config class directly to set up a lightweight demonstration environment.
from library.config import Config

# Set specific paths for this demo to avoid interfering with main results
DEMO_DIR = "./working/demo_execution"
if os.path.exists(DEMO_DIR):
    shutil.rmtree(DEMO_DIR)
os.makedirs(DEMO_DIR, exist_ok=True)

Config.WORKING_DIR = DEMO_DIR
Config.VOCAB_PATH = os.path.join(DEMO_DIR, "vocab.parquet")
Config.EMBEDDING_MATRIX_PATH = os.path.join(DEMO_DIR, "embedding_matrix.npy")
Config.RANKER_TRAIN_DATA = os.path.join(DEMO_DIR, "ranker_train_data.parquet")
Config.RANKER_VAL_DATA = os.path.join(DEMO_DIR, "ranker_val_data.parquet")
Config.READER_TRAIN_DATA = os.path.join(DEMO_DIR, "reader_train_data.parquet")
Config.READER_VAL_DATA = os.path.join(DEMO_DIR, "reader_val_data.parquet")
Config.RANKER_MODEL_PATH = os.path.join(DEMO_DIR, "ranker_best.pth")
Config.READER_MODEL_PATH = os.path.join(DEMO_DIR, "reader_best.pth")
Config.SUBMISSION_DIR = os.path.join(DEMO_DIR, "submission")
Config.SUBMISSION_FILE = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Hyperparameters for speed
Config.DEBUG = True
Config.SAMPLE_SIZE = 50  # Very small sample for demonstration
Config.BATCH_SIZE = 4
Config.NUM_EPOCHS = 1
Config.VOCAB_SIZE = 2000
Config.EMBEDDING_DIM = 50  # Smaller embeddings for speed
Config.RANKER_HIDDEN_DIM = 32
Config.READER_HIDDEN_DIM = 32

# Ensure directories exist
Config.setup()

# Set seeds for reproducibility
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(Config.SEED)

# Import library modules after configuring Config
from library.text_utils import TextProcessor, Vocabulary, get_vocab_and_matrix
from library.data_loader import (
    get_data_loaders,
    create_ranker_data,
    create_reader_data,
    create_inference_data,
    RankerDataset,
    ReaderDataset,
    collate_ranker,
    collate_reader,
)
from library.ranker_model import DIPNRanker
from library.reader_model import QIRNReader
from library.trainer import ModelTrainer
from library.inference import PredictionPipeline


def main():
    print("=== Starting Demonstration Pipeline ===")

    # --------------------------------------------------------------------------
    # 2. Text Utilities Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] TextProcessor and Vocabulary")
    processor = TextProcessor()

    sample_text = "<P> This is a sample document. </P> <Ul> <Li> Item 1 </Li> </Ul>"
    tokens = processor.tokenize(sample_text)
    segments = processor.segment_document(sample_text)

    print(f"Tokens: {tokens[:5]}...")
    print(f"Segments: {len(segments)}")
    assert len(tokens) > 0, "Tokenization failed"
    assert len(segments) == 2, f"Segmentation expected 2 parts, got {len(segments)}"

    # Build Vocabulary manually to ensure we have one for the demo
    # In a real run, we might read from files, here we simulate some text
    dummy_texts = [
        "what is the capital of france",
        "paris is the capital of france",
        "the quick brown fox jumps over the lazy dog",
        "<P> html tags should be handled </P>",
    ]

    vocab = Vocabulary()
    vocab.build(dummy_texts, max_vocab_size=Config.VOCAB_SIZE)
    vocab.save()

    # Verify Vocab
    assert vocab.vocab_size > 2, "Vocabulary size too small"
    assert Config.PAD_TOKEN in vocab.token_to_idx

    # Create Embeddings
    embedding_matrix = vocab.create_embedding_matrix(load_cached_data=False)
    assert embedding_matrix.shape == (vocab.vocab_size, Config.EMBEDDING_DIM)
    print("Vocabulary and Embeddings created successfully.")

    # --------------------------------------------------------------------------
    # 3. Data Loader Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Data Loading")

    # We will use the provided metadata files but Config.SAMPLE_SIZE limits the processing
    # Note: create_ranker_data reads actual input files. Since input is read-only and valid,
    # this works.

    # Create Datasets
    # We force load_cached_data=False to trigger processing logic
    ranker_train_df = create_ranker_data(
        Config.TRAIN_METADATA, vocab, load_cached_data=False
    )
    reader_train_df = create_reader_data(
        Config.TRAIN_METADATA, vocab, load_cached_data=False
    )

    print(f"Ranker Train Samples: {len(ranker_train_df)}")
    print(f"Reader Train Samples: {len(reader_train_df)}")

    # If no data found (due to small sample size filtering), create dummy data for model verification
    if len(ranker_train_df) == 0:
        print("Warning: No ranker data found in sample. Creating dummy data.")
        ranker_train_df = pd.DataFrame(
            {
                "q_indices": [[2, 3, 4]] * 10,
                "p_indices": [[2, 3, 4, 5, 6]] * 10,
                "label": [1, 0] * 5,
            }
        )

    if len(reader_train_df) == 0:
        print("Warning: No reader data found in sample. Creating dummy data.")
        reader_train_df = pd.DataFrame(
            {
                "q_indices": [[2, 3, 4]] * 10,
                "p_indices": [[2, 3, 4, 5, 6, 7, 8]] * 10,
                "start_token": [0] * 10,
                "end_token": [2] * 10,
            }
        )

    # Instantiate Datasets and Loaders
    ranker_ds = RankerDataset(ranker_train_df)
    reader_ds = ReaderDataset(reader_train_df)

    ranker_loader = torch.utils.data.DataLoader(
        ranker_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_ranker
    )
    reader_loader = torch.utils.data.DataLoader(
        reader_ds, batch_size=Config.BATCH_SIZE, collate_fn=collate_reader
    )

    # Verify Batch Shapes
    q_batch, p_batch, l_batch = next(iter(ranker_loader))
    print(f"Ranker Batch - Q: {q_batch.shape}, P: {p_batch.shape}, L: {l_batch.shape}")
    assert q_batch.dim() == 2
    assert p_batch.dim() == 2
    assert l_batch.dim() == 1

    q_r_batch, p_r_batch, start_batch, end_batch = next(iter(reader_loader))
    print(
        f"Reader Batch - Q: {q_r_batch.shape}, P: {p_r_batch.shape}, Start: {start_batch.shape}"
    )
    assert start_batch.dim() == 1

    # --------------------------------------------------------------------------
    # 4. Model Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Model Initialization and Forward Pass")

    # Ranker
    ranker_model = DIPNRanker(embedding_matrix).to(Config.DEVICE)
    ranker_out = ranker_model(q_batch.to(Config.DEVICE), p_batch.to(Config.DEVICE))
    print(f"Ranker Output Shape: {ranker_out.shape}")
    assert ranker_out.shape == (q_batch.size(0),), "Ranker output shape mismatch"

    # Reader
    reader_model = QIRNReader(embedding_matrix).to(Config.DEVICE)
    start_logits, end_logits = reader_model(
        q_r_batch.to(Config.DEVICE), p_r_batch.to(Config.DEVICE)
    )
    print(f"Reader Output Shapes: Start {start_logits.shape}, End {end_logits.shape}")
    assert start_logits.shape == p_r_batch.shape, "Reader start logits shape mismatch"
    assert end_logits.shape == p_r_batch.shape, "Reader end logits shape mismatch"

    # --------------------------------------------------------------------------
    # 5. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Training Loop")
    trainer = ModelTrainer()

    # Train Ranker (1 epoch)
    # We use the same loader for train and val for simplicity in this demo
    trained_ranker = trainer.train_ranker(
        ranker_loader, ranker_loader, embedding_matrix
    )
    assert isinstance(trained_ranker, DIPNRanker)

    # Train Reader (1 epoch)
    trained_reader = trainer.train_reader(
        reader_loader, reader_loader, embedding_matrix
    )
    assert isinstance(trained_reader, QIRNReader)

    # --------------------------------------------------------------------------
    # 6. Inference Demonstration
    # --------------------------------------------------------------------------
    print("\n[Demo] Inference Pipeline")

    # Create a dummy test metadata file for inference demonstration
    # This ensures we don't process the huge full test set
    demo_test_meta_path = os.path.join(DEMO_DIR, "demo_test_metadata.csv")
    test_meta_df = pd.read_csv(Config.TEST_METADATA).head(20)  # Take first 20 lines
    test_meta_df.to_csv(demo_test_meta_path, index=False)

    pipeline = PredictionPipeline()

    # Manually inject resources to avoid reloading from disk
    pipeline.device = Config.DEVICE
    pipeline.vocab = vocab
    pipeline.embedding_matrix = embedding_matrix
    pipeline.ranker = trained_ranker
    pipeline.reader = trained_reader

    # Run inference
    results_df = pipeline.run_inference(test_metadata_path=demo_test_meta_path)

    print("Inference Results:")
    print(results_df.head())

    assert "example_id" in results_df.columns
    assert "long_answer" in results_df.columns
    assert "short_answer" in results_df.columns

    # Generate Submission
    pipeline.generate_submission(results_df)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file not created"
    print(f"Submission file created at {Config.SUBMISSION_FILE}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()

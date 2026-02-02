import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil
from torch.utils.data import DataLoader

# Import library components
from library.config import Config
import importlib
import library.preprocessing

importlib.reload(library.preprocessing)
from library.preprocessing import get_tokenizer, get_embedding_matrix, HTMLParser
from library.dataset import (
    get_ranker_dataset,
    get_reader_dataset,
    ranker_collate_fn,
    reader_collate_fn,
)
from library.models import HistogramRanker, QRNNReader
from library.trainer import RankerTrainer, ReaderTrainer
from library.inference import InferencePipeline


def setup_demo_config():
    """
    Overrides Config parameters to create a lightweight execution environment.
    """
    # Define a separate working directory for this demo
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)

    # Update Paths
    Config.WORKING_DIR = demo_dir
    Config.VOCAB_PATH = os.path.join(demo_dir, "vocab.parquet")
    Config.EMBEDDING_MATRIX_PATH = os.path.join(demo_dir, "embedding_matrix.npy")
    Config.RANKER_TRAIN_DATA = os.path.join(demo_dir, "ranker_train_data.parquet")
    Config.RANKER_VAL_DATA = os.path.join(demo_dir, "ranker_val_data.parquet")
    Config.READER_TRAIN_DATA = os.path.join(demo_dir, "reader_train_data.parquet")
    Config.READER_VAL_DATA = os.path.join(demo_dir, "reader_val_data.parquet")
    Config.TEST_FEATURES_PATH = os.path.join(demo_dir, "ranker_test_features.parquet")
    Config.RANKER_MODEL_PATH = os.path.join(demo_dir, "ranker_best.pth")
    Config.READER_MODEL_PATH = os.path.join(demo_dir, "reader_best.pth")
    Config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    Config.FINAL_SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Ensure submission dir exists
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    # Update Hyperparameters for Speed
    Config.DEBUG = True
    Config.TRAIN_SUBSET_SIZE = 500  # Small subset for training
    Config.VAL_SUBSET_SIZE = 100  # Small subset for validation
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.VOCAB_SIZE = 1000  # Small vocab
    Config.EMBEDDING_DIM = 32  # Small embedding dim
    Config.HIDDEN_DIM = 32
    Config.QRNN_HIDDEN_DIM = 32
    Config.RANKER_HIDDEN_DIM = 32

    # Create a small test metadata file to speed up inference
    # The original get_test_dataset processes the whole file defined in Config.TEST_METADATA_PATH
    full_test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
    demo_test_meta_path = os.path.join(demo_dir, "demo_test_metadata.csv")
    full_test_meta.head(20).to_csv(demo_test_meta_path, index=False)
    Config.TEST_METADATA_PATH = demo_test_meta_path

    print(f"Config updated. Working directory: {Config.WORKING_DIR}")


def set_seeds(seed=42):
    """Sets random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def demo_preprocessing():
    print("\n=== 1. Preprocessing Demo ===")

    # 1.1 HTML Parsing
    parser = HTMLParser()
    dummy_html = (
        "<P> This is a paragraph. </P> <H1> Header </H1> <Ul> <Li> Item 1 </Li> </Ul>"
    )
    segments = parser.segment(dummy_html)
    print(f"HTML Parsing: {len(segments)} segments found.")
    assert len(segments) == 3, "HTMLParser failed to segment correctly"
    assert (
        segments[0]["text"] == "<P> This is a paragraph. </P>"
    ), "Segment text mismatch"

    # 1.2 Tokenizer Construction
    # We force build from scratch using the small training subset defined in Config
    print("Building Tokenizer from scratch...")
    tokenizer = get_tokenizer(load_cached_data=False)
    assert tokenizer.is_fitted, "Tokenizer should be fitted"
    print(f"Tokenizer vocab size: {len(tokenizer.token_to_id)}")

    # 1.3 Embedding Matrix
    print("Creating Embedding Matrix...")
    embedding_matrix = get_embedding_matrix(tokenizer, load_cached_data=False)
    assert embedding_matrix.shape == (len(tokenizer.token_to_id), Config.EMBEDDING_DIM)
    print(f"Embedding matrix shape: {embedding_matrix.shape}")

    return tokenizer, embedding_matrix


def demo_datasets(tokenizer):
    print("\n=== 2. Dataset & Dataloader Demo ===")

    # 2.1 Ranker Dataset
    print("Generating Ranker Dataset (Train)...")
    ranker_train_ds = get_ranker_dataset(
        "train", tokenizer, load_cached_data=False, debug=True
    )
    print(f"Ranker Train Samples: {len(ranker_train_ds)}")

    if len(ranker_train_ds) > 0:
        ranker_loader = DataLoader(
            ranker_train_ds, batch_size=Config.BATCH_SIZE, collate_fn=ranker_collate_fn
        )
        batch = next(iter(ranker_loader))
        q, pos, neg = batch
        print(f"Ranker Batch Shapes - Q: {q.shape}, Pos: {pos.shape}, Neg: {neg.shape}")
        assert q.shape[0] == pos.shape[0] == neg.shape[0]
    else:
        print(
            "Warning: Ranker dataset is empty (likely due to small subset). Skipping loader check."
        )

    # 2.2 Reader Dataset
    print("Generating Reader Dataset (Train)...")
    reader_train_ds = get_reader_dataset(
        "train", tokenizer, load_cached_data=False, debug=True
    )
    print(f"Reader Train Samples: {len(reader_train_ds)}")

    if len(reader_train_ds) > 0:
        reader_loader = DataLoader(
            reader_train_ds, batch_size=Config.BATCH_SIZE, collate_fn=reader_collate_fn
        )
        batch = next(iter(reader_loader))
        inp, start, end = batch
        print(
            f"Reader Batch Shapes - Input: {inp.shape}, Start: {start.shape}, End: {end.shape}"
        )
        assert inp.shape[0] == start.shape[0] == end.shape[0]
    else:
        print("Warning: Reader dataset is empty. Skipping loader check.")

    return ranker_train_ds, reader_train_ds


def demo_models_and_training(tokenizer, embedding_matrix, ranker_ds, reader_ds):
    print("\n=== 3. Models & Training Demo ===")

    device = Config.DEVICE

    # --- Ranker ---
    print("Initializing HistogramRanker...")
    ranker = HistogramRanker(embedding_matrix).to(device)

    # Forward pass check
    dummy_q = torch.randint(0, Config.VOCAB_SIZE, (2, 10)).to(device)
    dummy_d = torch.randint(0, Config.VOCAB_SIZE, (2, 50)).to(device)
    scores = ranker(dummy_q, dummy_d)
    assert scores.shape == (2,), f"Ranker output shape mismatch: {scores.shape}"

    # Training Loop
    if len(ranker_ds) > 0:
        print("Running Ranker Training Loop...")
        ranker_loader = DataLoader(
            ranker_ds, batch_size=Config.BATCH_SIZE, collate_fn=ranker_collate_fn
        )
        # We reuse training set as validation set just for the demo to ensure non-empty val
        trainer = RankerTrainer(ranker, device=device)
        trainer.fit(ranker_loader, ranker_loader, epochs=1)

    # --- Reader ---
    print("Initializing QRNNReader...")
    reader = QRNNReader(embedding_matrix).to(device)

    # Forward pass check
    dummy_inp = torch.randint(0, Config.VOCAB_SIZE, (2, 60)).to(device)
    s_logits, e_logits = reader(dummy_inp)
    assert s_logits.shape == (
        2,
        60,
    ), f"Reader start logits shape mismatch: {s_logits.shape}"
    assert e_logits.shape == (
        2,
        60,
    ), f"Reader end logits shape mismatch: {e_logits.shape}"

    # Training Loop
    if len(reader_ds) > 0:
        print("Running Reader Training Loop...")
        reader_loader = DataLoader(
            reader_ds, batch_size=Config.BATCH_SIZE, collate_fn=reader_collate_fn
        )
        trainer = ReaderTrainer(reader, device=device)
        trainer.fit(reader_loader, reader_loader, epochs=1)


def demo_inference():
    print("\n=== 4. Inference Pipeline Demo ===")

    # Initialize pipeline (this reloads the models saved during training if they exist)
    # Note: If training didn't save a model (e.g. dataset empty), it warns and uses random weights.
    pipeline = InferencePipeline(load_cached_data=True)

    print("Generating Submission...")
    pipeline.generate_submission()

    if os.path.exists(Config.FINAL_SUBMISSION_PATH):
        df = pd.read_csv(Config.FINAL_SUBMISSION_PATH)
        print(f"Submission generated successfully. Rows: {len(df)}")
        print(df.head())

        # Validation: check format
        assert "example_id" in df.columns
        assert "PredictionString" in df.columns
    else:
        print("Error: Submission file was not created.")


if __name__ == "__main__":
    # 1. Setup
    set_seeds(42)
    setup_demo_config()

    # 2. Preprocessing
    tokenizer, embedding_matrix = demo_preprocessing()

    # 3. Datasets
    ranker_ds, reader_ds = demo_datasets(tokenizer)

    # 4. Models & Training
    demo_models_and_training(tokenizer, embedding_matrix, ranker_ds, reader_ds)

    # 5. Inference
    demo_inference()

    print("\n=== Demo Completed Successfully ===")

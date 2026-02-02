import os
import shutil
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import library modules
from library import config
from library import text_utils
from library import data_factory
from library import layers
from library import ranker_net
from library import reader_net
from library import training_utils
from library import inference_utils


def setup_demo_environment():
    """Sets up a temporary directory and overrides config for a fast demo run."""
    print("--- Setting up Demo Environment ---")
    demo_dir = "./working/demo_execution"
    if os.path.exists(demo_dir):
        shutil.rmtree(demo_dir)
    os.makedirs(demo_dir, exist_ok=True)
    os.makedirs(os.path.join(demo_dir, "submission"), exist_ok=True)

    # Override config paths
    config.WORKING_DIR = demo_dir
    config.VOCAB_CACHE_PATH = os.path.join(demo_dir, "vocab.parquet")
    config.EMBEDDING_MATRIX_CACHE_PATH = os.path.join(demo_dir, "embedding_matrix.npy")
    config.RANKER_TRAIN_CACHE = os.path.join(demo_dir, "ranker_train_data.parquet")
    config.RANKER_VAL_CACHE = os.path.join(demo_dir, "ranker_val_data.parquet")
    config.READER_TRAIN_CACHE = os.path.join(demo_dir, "reader_train_data.parquet")
    config.READER_VAL_CACHE = os.path.join(demo_dir, "reader_val_data.parquet")
    config.RANKER_TEST_INPUTS_CACHE = os.path.join(
        demo_dir, "ranker_test_inputs.parquet"
    )
    config.RANKER_MODEL_PATH = os.path.join(demo_dir, "ranker_best.pth")
    config.READER_MODEL_PATH = os.path.join(demo_dir, "reader_best.pth")
    config.SUBMISSION_DIR = os.path.join(demo_dir, "submission")
    config.SUBMISSION_PATH = os.path.join(config.SUBMISSION_DIR, "submission.csv")

    # Use a small subset of test metadata for the demo inference
    # We create a small test metadata file in the demo dir
    demo_test_meta_path = os.path.join(demo_dir, "demo_test_metadata.csv")
    full_test_meta = pd.read_csv(config.TEST_METADATA_PATH)
    full_test_meta.head(20).to_csv(demo_test_meta_path, index=False)
    config.TEST_METADATA_PATH = demo_test_meta_path

    # Override hyperparameters for speed
    config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples
    config.MAX_VOCAB_SIZE = 500  # Small vocab
    config.EMBEDDING_DIM = 32  # Small embeddings
    config.HIDDEN_DIM = 32  # Small hidden state
    config.BATCH_SIZE = 4  # Small batch size
    config.NUM_EPOCHS = 1  # 1 Epoch
    config.MAX_Q_LEN = 10  # Short sequences
    config.MAX_DOC_LEN = 50  # Short sequences
    config.K_MAX = 3  # K-Max pooling size

    print(f"Config overrides applied. Working dir: {config.WORKING_DIR}")


def test_text_utils():
    """Verifies vocabulary building and embedding loading."""
    print("\n--- Testing Text Utils ---")

    # Build Vocab
    vocab = text_utils.build_vocab(load_cached_data=False)
    print(f"Vocabulary built. Size: {len(vocab)}")
    assert len(vocab) > 4, "Vocabulary should contain at least special tokens"
    assert "<PAD>" in vocab
    assert vocab["<PAD>"] == 0

    # Load Embeddings
    embeddings = text_utils.load_embeddings(vocab, load_cached_data=False)
    print(f"Embeddings loaded. Shape: {embeddings.shape}")
    assert embeddings.shape == (len(vocab), config.EMBEDDING_DIM)
    assert np.all(embeddings[0] == 0), "Padding embedding should be zero"

    return vocab, embeddings


def test_data_factory(vocab):
    """Verifies data processing for Ranker and Reader."""
    print("\n--- Testing Data Factory ---")

    # Process Ranker Data
    print("Processing Ranker Data...")
    ranker_df = data_factory.process_ranker_data(
        config.TRAIN_METADATA_PATH, vocab, load_cached_data=False, is_train=True
    )
    print(f"Ranker Data Rows: {len(ranker_df)}")
    if len(ranker_df) > 0:
        assert "q_indices" in ranker_df.columns
        assert "pos_indices" in ranker_df.columns
        assert "neg_indices" in ranker_df.columns
        # Check sequence lengths
        assert len(ranker_df.iloc[0]["q_indices"]) == config.MAX_Q_LEN
        assert len(ranker_df.iloc[0]["pos_indices"]) == config.MAX_DOC_LEN

    # Process Reader Data
    print("Processing Reader Data...")
    reader_df = data_factory.process_reader_data(
        config.TRAIN_METADATA_PATH, vocab, load_cached_data=False, is_train=True
    )
    print(f"Reader Data Rows: {len(reader_df)}")
    if len(reader_df) > 0:
        assert "q_indices" in reader_df.columns
        assert "para_indices" in reader_df.columns
        assert "start_idx" in reader_df.columns
        assert "end_idx" in reader_df.columns

    return ranker_df, reader_df


def test_layers():
    """Verifies custom neural network layers."""
    print("\n--- Testing Layers ---")
    batch_size = 2
    seq_len = 10
    dim = config.EMBEDDING_DIM

    # Test Highway Layer
    highway = layers.HighwayLayer(dim)
    x = torch.randn(batch_size, seq_len, dim)
    out = highway(x)
    assert out.shape == x.shape, f"Highway output shape mismatch: {out.shape}"
    print("HighwayLayer check passed.")

    # Test CoAttention
    coattn = layers.CoAttention(dim)
    q = torch.randn(batch_size, 5, dim)  # Query len 5
    c = torch.randn(batch_size, 10, dim)  # Context len 10
    out = coattn(q, c)
    # Output should be (Batch, Context_Len, 2*Dim)
    expected_shape = (batch_size, 10, 2 * dim)
    assert (
        out.shape == expected_shape
    ), f"CoAttention output shape mismatch: {out.shape}"
    print("CoAttention check passed.")

    # Test KMaxPooling
    k = config.K_MAX
    kmax = layers.KMaxPooling(k=k)
    q = torch.randn(batch_size, 5, dim)
    c = torch.randn(batch_size, 10, dim)
    out = kmax(q, c)
    # Output should be (Batch, K)
    expected_shape = (batch_size, k)
    assert (
        out.shape == expected_shape
    ), f"KMaxPooling output shape mismatch: {out.shape}"
    print("KMaxPooling check passed.")


def test_models_and_training(vocab, embeddings, ranker_df, reader_df):
    """Verifies model instantiation and training loops."""
    print("\n--- Testing Models and Training ---")

    device = torch.device(
        "cpu"
    )  # Use CPU for demo to avoid CUDA overhead/memory issues on small data
    vocab_size = len(vocab)

    # --- Ranker ---
    print("Testing Ranker Model...")
    ranker = ranker_net.KMaxInteractionRanker(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_embeddings=embeddings,
        k=config.K_MAX,
        hidden_dim=config.HIDDEN_DIM,
    ).to(device)

    if len(ranker_df) > 0:
        # Create DataLoader
        ranker_ds = data_factory.RankerDataset(ranker_df)
        ranker_loader = DataLoader(ranker_ds, batch_size=config.BATCH_SIZE)

        # Run Training Loop (1 epoch)
        training_utils.train_ranker(
            ranker, ranker_loader, ranker_loader, device, epochs=1
        )
        assert os.path.exists(
            config.RANKER_MODEL_PATH
        ), "Ranker model checkpoint not saved."
    else:
        print("Skipping Ranker training (no data generated).")

    # --- Reader ---
    print("Testing Reader Model...")
    reader = reader_net.HighwayCoAttentionReader(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        pretrained_embeddings=embeddings,
        hidden_dim=config.HIDDEN_DIM,
        num_highway_layers=1,
    ).to(device)

    if len(reader_df) > 0:
        # Create DataLoader
        reader_ds = data_factory.ReaderDataset(reader_df)
        reader_loader = DataLoader(reader_ds, batch_size=config.BATCH_SIZE)

        # Run Training Loop (1 epoch)
        training_utils.train_reader(
            reader, reader_loader, reader_loader, device, epochs=1
        )
        assert os.path.exists(
            config.READER_MODEL_PATH
        ), "Reader model checkpoint not saved."
    else:
        print("Skipping Reader training (no data generated).")


def test_inference_pipeline():
    """Verifies the end-to-end inference pipeline."""
    print("\n--- Testing Inference Pipeline ---")

    pipeline = inference_utils.InferencePipeline()

    # We rely on the models trained/saved in the previous step
    # If no data was generated, dummy weights will be initialized by the pipeline

    try:
        pipeline.run_inference()

        # Verify submission file
        if os.path.exists(config.SUBMISSION_PATH):
            sub_df = pd.read_csv(config.SUBMISSION_PATH)
            print(f"Submission generated with {len(sub_df)} rows.")
            print(sub_df.head())

            # Basic format check
            assert "example_id" in sub_df.columns
            assert "PredictionString" in sub_df.columns
        else:
            raise FileNotFoundError("Submission file was not created.")

    except Exception as e:
        print(f"Inference failed: {e}")
        raise e


if __name__ == "__main__":
    # 1. Setup
    setup_demo_environment()

    # 2. Text Utils
    vocab, embeddings = test_text_utils()

    # 3. Data Processing
    ranker_df, reader_df = test_data_factory(vocab)

    # 4. Layers
    test_layers()

    # 5. Models & Training
    test_models_and_training(vocab, embeddings, ranker_df, reader_df)

    # 6. Inference
    test_inference_pipeline()

    print("\n--- Demo Completed Successfully ---")

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

# Import from the provided library
from library.config import Config
from library.utils import set_seed
from library.data_processing import DataProcessor
from library.dataset import NQDataset
from library.model import FeedForwardDecomposableAttention
from library.trainer import Trainer
from library.inference import InferenceManager


def run_demonstration():
    print("--- Starting Demonstration ---")

    # 1. Configuration Setup
    # We override defaults to ensure the demo runs quickly within the time limit
    config = Config()
    config.DEBUG_SAMPLE_SIZE = 50  # Process only 50 samples
    config.NUM_EPOCHS = 1  # Train for only 1 epoch
    config.BATCH_SIZE = 4  # Small batch size
    config.EMBED_DIM = 32  # Smaller embedding dimension
    config.VOCAB_SIZE = 1000  # Smaller vocabulary
    config.HIDDEN_DIM = 64  # Smaller hidden layer

    # Ensure reproducibility
    set_seed(config.SEED)
    config.display()

    # 2. Data Processing
    print("\n--- Data Processing ---")
    processor = DataProcessor(config)

    # Build Vocabulary
    # Note: This will read the first 50 samples of the train file due to DEBUG_SAMPLE_SIZE
    vocab = processor.build_vocab(load_cached_data=False)
    assert len(vocab) > 2, "Vocabulary should contain at least PAD and UNK tokens"
    assert config.PAD_TOKEN in vocab, "PAD token missing from vocab"
    assert config.UNK_TOKEN in vocab, "UNK token missing from vocab"
    print(f"Vocab built successfully. Size: {len(vocab)}")

    # Create Embedding Matrix
    embedding_matrix = processor.create_embedding_matrix(load_cached_data=False)
    assert embedding_matrix.shape == (
        len(vocab),
        config.EMBED_DIM,
    ), f"Embedding matrix shape mismatch. Expected {(len(vocab), config.EMBED_DIM)}, got {embedding_matrix.shape}"
    print("Embedding matrix created successfully.")

    # 3. Dataset and DataLoader
    print("\n--- Dataset Loading ---")
    # We disable loading cached parquet files to force the dataset to use our current debug config
    train_dataset = NQDataset(config, processor, split="train", load_cached_data=False)
    val_dataset = NQDataset(config, processor, split="val", load_cached_data=False)

    # Verify Dataset Length (should be <= DEBUG_SAMPLE_SIZE, usually exactly equal unless filtered completely)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")
    assert len(train_dataset) <= config.DEBUG_SAMPLE_SIZE

    # Verify __getitem__ structure
    sample = train_dataset[0]
    required_keys = [
        "q_indices",
        "c_indices",
        "long_label",
        "short_start",
        "short_end",
        "yn_label",
    ]
    for key in required_keys:
        assert key in sample, f"Sample missing key: {key}"

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=NQDataset.collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=NQDataset.collate_fn,
    )

    # Verify Batch Structure
    batch = next(iter(train_loader))
    assert "q_input" in batch
    assert (
        batch["q_input"].shape[1] == config.Q_MAX_LEN
    ), "Incorrect Question Sequence Length"
    assert (
        batch["c_input"].shape[1] == config.C_MAX_LEN
    ), "Incorrect Candidate Sequence Length"
    print("DataLoaders initialized and verified.")

    # 4. Model Initialization
    print("\n--- Model Initialization ---")
    embedding_tensor = torch.tensor(embedding_matrix, dtype=torch.float32)
    model = FeedForwardDecomposableAttention(config, embedding_tensor)

    # Forward Pass Verification
    model.to(config.DEVICE)
    # Move batch to device
    q_input = batch["q_input"].to(config.DEVICE)
    c_input = batch["c_input"].to(config.DEVICE)

    outputs = model(q_input, c_input)

    # Check output shapes
    batch_dim = q_input.shape[0]
    assert outputs["ranking_logits"].shape == (
        batch_dim,
        1,
    ), "Ranking logits shape mismatch"
    assert outputs["start_logits"].shape == (
        batch_dim,
        config.C_MAX_LEN,
    ), "Start logits shape mismatch"
    assert outputs["end_logits"].shape == (
        batch_dim,
        config.C_MAX_LEN,
    ), "End logits shape mismatch"
    assert outputs["yn_logits"].shape == (
        batch_dim,
        config.NUM_CLASSES_YN,
    ), "Yes/No logits shape mismatch"
    print("Model forward pass verified.")

    # 5. Training
    print("\n--- Training Loop ---")
    trainer = Trainer(model, train_loader, val_loader, config)
    trainer.train()

    # Verify checkpoint creation
    assert os.path.exists(
        config.MODEL_CHECKPOINT_PATH
    ), "Model checkpoint was not saved."
    print("Training completed and checkpoint verified.")

    # 6. Inference
    print("\n--- Inference ---")
    # Initialize InferenceManager
    # It re-initializes processor/model internally, but loads the checkpoint we just saved.
    inference_manager = InferenceManager(config)

    # Run predictions on a small subset of test data
    submission_df = inference_manager.generate_predictions(debug_sample_size=10)

    # Verify Submission DataFrame
    assert not submission_df.empty, "Submission DataFrame is empty"
    assert "example_id" in submission_df.columns, "Missing example_id column"
    assert (
        "PredictionString" in submission_df.columns
    ), "Missing PredictionString column"

    # Check format of example_id (should end in _long or _short)
    example_id = submission_df.iloc[0]["example_id"]
    assert example_id.endswith("_long") or example_id.endswith(
        "_short"
    ), f"Invalid example_id format: {example_id}"

    print("Inference generated successfully.")
    print(f"Submission shape: {submission_df.shape}")
    print("Sample predictions:")
    print(submission_df.head())

    # Save submission (mock run)
    inference_manager.save_submission(submission_df)
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file not found on disk."

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demonstration()

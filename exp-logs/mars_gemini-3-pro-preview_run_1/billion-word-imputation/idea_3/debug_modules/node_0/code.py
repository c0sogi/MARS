import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import Config
from library.utils import get_or_build_vocab, SOS_TOKEN, EOS_TOKEN
from library.dataset import GapDataset, get_dataloaders, collate_fn
from library.model import DecoupledTransformer
from library.trainer import Trainer
from library.inference import InferenceEngine

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def setup_demo_config():
    """
    Overrides default configuration for a fast demonstration run.
    """
    print("--- Configuring for Demo ---")

    # Paths
    Config.WORK_DIR = "./working/demo_execution"
    Config.MODEL_PATH = os.path.join(Config.WORK_DIR, "best_model.pth")
    Config.VOCAB_PATH = os.path.join(Config.WORK_DIR, "vocab.npy")
    Config.SUBMISSION_PATH = os.path.join(Config.WORK_DIR, "submission.csv")

    # Ensure working directory exists
    os.makedirs(Config.WORK_DIR, exist_ok=True)

    # Data Constraints
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 200  # Small subset for speed
    Config.VOCAB_SIZE = 1000  # Small vocab for speed
    Config.MAX_LEN = 32  # Short sequences
    Config.BATCH_SIZE = 8

    # Model Constraints (Tiny model)
    Config.D_MODEL = 64
    Config.N_LAYERS = 2
    Config.N_HEADS = 2
    Config.DIM_FEEDFORWARD = 128

    # Training Constraints
    Config.MAX_EPOCHS = 1
    Config.LEARNING_RATE = 1e-3
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Set Seed
    Config.set_seed(42)
    print("Configuration updated for speed and reproducibility.")


def demonstrate_vocabulary():
    """
    Demonstrates vocabulary building and validation.
    """
    print("\n--- Demonstrating Vocabulary ---")

    # Build vocab (will use DEBUG_SAMPLE_SIZE rows)
    vocab = get_or_build_vocab(load_cached_data=False)

    print(f"Vocabulary size: {len(vocab)}")

    # Assertions
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert Config.PAD_TOKEN in vocab.stoi, "PAD token missing"
    assert Config.UNK_TOKEN in vocab.stoi, "UNK token missing"
    assert SOS_TOKEN in vocab.stoi, "SOS token missing"
    assert EOS_TOKEN in vocab.stoi, "EOS token missing"

    # Encode/Decode check
    test_tokens = ["the", "quick", "brown", "fox"]
    indices = vocab.encode(test_tokens)
    decoded = vocab.decode(indices)

    assert len(indices) == len(test_tokens), "Encoding length mismatch"
    assert len(decoded) == len(test_tokens), "Decoding length mismatch"

    print("Vocabulary validation passed.")
    return vocab


def demonstrate_dataset(vocab):
    """
    Demonstrates Dataset and DataLoader functionality.
    """
    print("\n--- Demonstrating Dataset & DataLoader ---")

    # Instantiate Dataset directly
    ds = GapDataset(Config.TRAIN_METADATA, vocab, mode="train", load_cached_data=False)

    assert len(ds) > 0, "Dataset should not be empty"

    # Check single item
    input_ids, gap_idx, target_id = ds[0]

    # Validation logic
    # input_ids should be a tensor of token indices
    assert isinstance(input_ids, torch.Tensor)
    # gap_idx should be a scalar tensor
    assert gap_idx.ndim == 0
    # target_id should be a scalar tensor
    assert target_id.ndim == 0

    # The gap index must be within the bounds of the input sequence
    # Note: gap_idx is the index *after* which the word was removed.
    # Valid range is roughly 0 to len(input_ids)-1
    assert (
        0 <= gap_idx.item() < len(input_ids)
    ), f"Gap index {gap_idx.item()} out of bounds for len {len(input_ids)}"

    print(
        f"Sample 0: Input Len={len(input_ids)}, Gap Index={gap_idx}, Target Token={target_id}"
    )

    # Check DataLoader batching
    loader, _, _ = get_dataloaders(vocab, load_cached_data=False)
    batch = next(iter(loader))

    assert "input_ids" in batch
    assert "gap_idx" in batch
    assert "target_id" in batch
    assert batch["input_ids"].shape[0] == Config.BATCH_SIZE

    print("Dataset and DataLoader validation passed.")


def demonstrate_model(vocab_size):
    """
    Demonstrates Model instantiation and forward pass.
    """
    print("\n--- Demonstrating Model Architecture ---")

    device = Config.DEVICE
    model = DecoupledTransformer(vocab_size).to(device)

    # Create dummy batch
    batch_size = 4
    seq_len = 16
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    dummy_mask = torch.ones((batch_size, seq_len)).to(device)

    # Forward pass
    loc_logits, id_logits = model(dummy_input, dummy_mask)

    # Check shapes
    # Loc logits: (Batch, Seq_Len)
    assert loc_logits.shape == (
        batch_size,
        seq_len,
    ), f"Loc logits shape mismatch: {loc_logits.shape}"

    # ID logits: (Batch, Seq_Len, Vocab_Size)
    assert id_logits.shape == (
        batch_size,
        seq_len,
        vocab_size,
    ), f"ID logits shape mismatch: {id_logits.shape}"

    print("Model forward pass validation passed.")


def demonstrate_training():
    """
    Demonstrates the Trainer class (Training loop).
    """
    print("\n--- Demonstrating Training Loop ---")

    trainer = Trainer()

    # Run fit (1 epoch, small data)
    print("Starting Trainer.fit()...")
    trainer.fit()

    # Check artifacts
    assert os.path.exists(Config.MODEL_PATH), "Model checkpoint was not saved."
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "Submission file was not generated by Trainer."

    print("Training loop completed successfully.")


def demonstrate_inference():
    """
    Demonstrates InferenceEngine explicitly.
    """
    print("\n--- Demonstrating Inference Engine ---")

    # Remove previous submission to ensure this run generates it
    if os.path.exists(Config.SUBMISSION_PATH):
        os.remove(Config.SUBMISSION_PATH)

    engine = InferenceEngine()
    engine.generate_submission(load_cached_data=True)  # Use cache generated by Trainer

    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), "InferenceEngine failed to generate submission."

    # Validate submission format
    df = pd.read_csv(Config.SUBMISSION_PATH)
    assert "id" in df.columns, "Submission missing 'id' column"
    assert "sentence" in df.columns, "Submission missing 'sentence' column"
    assert len(df) > 0, "Submission file is empty"

    # Check first row format
    first_sent = df.iloc[0]["sentence"]
    assert isinstance(first_sent, str), "Sentence is not a string"
    assert len(first_sent) > 0, "Sentence is empty"

    print(f"Submission generated with {len(df)} rows.")
    print("Inference validation passed.")


if __name__ == "__main__":
    # 1. Setup
    setup_demo_config()

    # 2. Vocabulary
    vocab = demonstrate_vocabulary()

    # 3. Dataset
    demonstrate_dataset(vocab)

    # 4. Model
    demonstrate_model(len(vocab))

    # 5. Training
    demonstrate_training()

    # 6. Inference
    demonstrate_inference()

    print("\n=== All Demonstrations Completed Successfully ===")

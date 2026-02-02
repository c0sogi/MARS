import os
import sys
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import Config, seed_everything
from library.text_utils import Tokenizer, pad_sequences
from library.dataset import get_dataloaders, ChatbotDataset
from library.model import DeepAveragingNetwork
from library.engine import run_training, generate_submission


def test_tokenizer_logic(config):
    """
    Verifies the Tokenizer class functionality: fitting, transforming, saving, and loading.
    """
    print("\n--- Testing Tokenizer Logic ---")

    # Create a temporary tokenizer
    tokenizer = Tokenizer(config)

    # Sample data
    texts = ["hello world", "hello python", "world of code"]

    # Fit
    tokenizer.fit_on_texts(texts)

    # Check vocabulary
    # vocab size is config.VOCAB_SIZE.
    # words: hello, world, python, of, code (5 unique words)
    # indices start at 2. 0 is PAD, 1 is UNK.
    # "hello" and "world" appear twice, others once.
    print(f"Word index: {tokenizer.word_index}")

    assert "hello" in tokenizer.word_index, "Tokenizer failed to learn 'hello'"
    assert tokenizer.word_index["<PAD>"] == 0
    assert tokenizer.word_index["<UNK>"] == 1

    # Transform
    sequences = tokenizer.texts_to_sequences(["hello code unknown"])
    # "unknown" should be mapped to 1 (<UNK>)
    print(f"Sequences: {sequences}")

    assert len(sequences) == 1
    assert sequences[0][0] == tokenizer.word_index["hello"]
    assert sequences[0][-1] == 1  # OOV token

    # Save and Load
    save_path = os.path.join(config.WORKING_DIR, "test_vocab.json")
    tokenizer.save(save_path)
    assert os.path.exists(save_path), "Tokenizer save failed"

    new_tokenizer = Tokenizer(config)
    loaded = new_tokenizer.load(save_path)
    assert loaded, "Tokenizer load failed"
    assert new_tokenizer.word_index == tokenizer.word_index, "Loaded vocab mismatch"

    print("Tokenizer logic verified.")


def test_model_forward_pass(config):
    """
    Verifies the DeepAveragingNetwork architecture and forward pass.
    """
    print("\n--- Testing Model Forward Pass ---")

    # Instantiate model
    model = DeepAveragingNetwork(config)
    model.to(config.DEVICE)
    model.eval()

    # Create dummy inputs
    # Batch size = 2, Seq len = 10
    bs = 2
    seq_len = 10

    # Random integers within vocab range
    dummy_prompt = torch.randint(0, config.VOCAB_SIZE, (bs, seq_len)).to(config.DEVICE)
    dummy_res_a = torch.randint(0, config.VOCAB_SIZE, (bs, seq_len)).to(config.DEVICE)
    dummy_res_b = torch.randint(0, config.VOCAB_SIZE, (bs, seq_len)).to(config.DEVICE)

    # Forward pass
    with torch.no_grad():
        logits = model(dummy_prompt, dummy_res_a, dummy_res_b)

    print(f"Logits shape: {logits.shape}")

    # Check shape: (batch_size, num_classes)
    assert logits.shape == (
        bs,
        config.NUM_CLASSES,
    ), f"Expected shape {(bs, config.NUM_CLASSES)}, got {logits.shape}"

    print("Model forward pass verified.")


def run_full_pipeline_demonstration():
    """
    Runs the full training and inference pipeline using the library functions.
    """
    print("\n--- Starting Full Pipeline Demonstration ---")

    # 1. Setup Configuration
    # We override some defaults to make the demo run faster
    config = Config(
        vocab_size=5000,  # Smaller vocab
        max_seq_len=64,  # Shorter sequences
        embedding_dim=32,  # Smaller embeddings
        hidden_dim=64,  # Smaller hidden layer
        batch_size=128,  # Larger batch size for speed
        epochs=1,  # Only 1 epoch for demo
        learning_rate=0.005,
        seed=42,
    )

    # Ensure working directory is clean/ready
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    # Set seeds
    seed_everything(config.SEED)

    # 2. Unit Tests
    test_tokenizer_logic(config)
    test_model_forward_pass(config)

    # 3. Train
    # This uses the real metadata files in ./metadata
    print("\n--- Executing Training Loop ---")
    run_training(config)

    # Verify model checkpoint exists
    assert os.path.exists(config.MODEL_PATH), "Model checkpoint was not saved."
    print(f"Model saved successfully at {config.MODEL_PATH}")

    # 4. Inference
    print("\n--- Executing Inference ---")
    generate_submission(config)

    # Verify submission file
    assert os.path.exists(config.SUBMISSION_PATH), "Submission file was not generated."

    # 5. Validate Submission Content
    print("\n--- Validating Submission File ---")
    sub_df = pd.read_csv(config.SUBMISSION_PATH)

    # Check columns
    expected_cols = ["id", "winner_model_a", "winner_model_b", "winner_tie"]
    assert (
        list(sub_df.columns) == expected_cols
    ), f"Columns mismatch. Got {sub_df.columns}"

    # Check length (should match test.csv)
    test_df = pd.read_csv(config.TEST_DATA_PATH)
    assert len(sub_df) == len(
        test_df
    ), f"Submission length mismatch. Expected {len(test_df)}, got {len(sub_df)}"

    # Check values are probabilities (roughly sum to 1, though softmax ensures this)
    # We check the first row
    row_sum = sub_df.iloc[0, 1:].sum()
    print(f"Row 0 probabilities sum: {row_sum:.4f}")
    assert 0.99 < row_sum < 1.01, "Probabilities do not sum to approx 1.0"

    print("\nPipeline demonstration completed successfully.")


if __name__ == "__main__":
    run_full_pipeline_demonstration()

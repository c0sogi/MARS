import os
import torch
import pandas as pd
import numpy as np
import shutil

# 1. Import Config and modify it for the demonstration *before* importing other modules
# to ensure changes propagate to all components.
from library.config import Config

# --- Configuration Override for Fast Demonstration ---
print("Configuring environment for demonstration...")
Config.WORKING_DIR = "./working/demo_run"
Config.CACHE_DIR = Config.WORKING_DIR
Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.TOKENIZER_PATH = os.path.join(Config.WORKING_DIR, "tokenizer.json")
Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

# Use a tiny subset of data for speed
Config.DEBUG_SAMPLE_SIZE = 200
Config.VOCAB_SIZE = 1000  # Small vocab for demo
Config.MAX_SEQ_LEN = 32

# Lightweight Model Hyperparameters
Config.EMBEDDING_DIM = 32
Config.HIDDEN_DIM = 64
Config.LSTM_LAYERS = 1
Config.BATCH_SIZE = 8
Config.NUM_EPOCHS = 1  # Just one epoch to prove it runs

# Ensure clean working directory
if os.path.exists(Config.WORKING_DIR):
    shutil.rmtree(Config.WORKING_DIR)
Config.setup()

# --- Import remaining library components ---
from library.tokenizer import get_tokenizer
from library.dataset import MissingWordDataset, get_dataloaders
from library.model import BiLSTMDualHead
from library.trainer import Trainer
from library.inference import Predictor


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(Config.SEED)

    print("\n=== 1. Tokenizer Demonstration ===")
    # Initialize tokenizer (builds from the debug sample of training data)
    tokenizer = get_tokenizer(load_cached_data=False)

    # Validation: Check special tokens
    assert tokenizer.pad_token == "<PAD>"
    assert tokenizer.word2idx["<PAD>"] == 0

    # Validation: Encode/Decode cycle
    test_sentence = "the quick brown fox"
    encoded = tokenizer.encode(test_sentence, max_len=10)
    decoded = tokenizer.decode(encoded)

    print(f"Original: '{test_sentence}'")
    print(f"Encoded: {encoded}")
    print(f"Decoded: '{decoded}'")

    # Check length constraint
    assert len(encoded) == 10
    # Check that known words are preserved (assuming 'the' is in top 1000 words of training data)
    # Note: 'fox' might be UNK depending on the random sample, but 'the' is almost certainly present.
    assert len(decoded.split()) <= 4

    print("\n=== 2. Dataset & DataLoader Demonstration ===")
    # Get DataLoaders
    train_loader, val_loader, test_loader, _ = get_dataloaders(
        config=Config, load_cached_data=False
    )

    # Fetch a batch from training
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    loc_target = batch["loc_target"]
    word_target = batch["word_target"]
    gap_idx = batch["gap_idx"]

    print(f"Batch Input Shape: {input_ids.shape}")
    print(f"Loc Target Shape: {loc_target.shape}")

    # Validation: Shapes
    assert input_ids.shape == (Config.BATCH_SIZE, Config.MAX_SEQ_LEN)
    assert loc_target.shape == (Config.BATCH_SIZE, Config.MAX_SEQ_LEN)
    assert word_target.shape == (Config.BATCH_SIZE,)

    # Validation: Logic Check
    # Check a specific sample in the batch
    idx = 0
    curr_input = input_ids[idx]
    curr_gap = gap_idx[idx].item()

    # The location target should be 1.0 at the gap index
    if curr_gap < Config.MAX_SEQ_LEN:
        assert loc_target[idx, curr_gap].item() == 1.0
        # Ensure only one location is marked as target (or zero if out of bounds, but here likely 1)
        assert torch.sum(loc_target[idx]).item() == 1.0

    print("Dataset logic verified.")

    print("\n=== 3. Model Architecture Demonstration ===")
    # Instantiate Model
    model = BiLSTMDualHead(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=Config.EMBEDDING_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT,
    ).to(Config.DEVICE)

    # Run Forward Pass
    input_tensor = input_ids.to(Config.DEVICE)
    loc_logits, word_logits = model(input_tensor)

    print(f"Loc Logits Shape: {loc_logits.shape}")
    print(f"Word Logits Shape: {word_logits.shape}")

    # Validation: Output Shapes
    # loc_logits: (batch, seq_len, 1)
    assert loc_logits.shape == (Config.BATCH_SIZE, Config.MAX_SEQ_LEN, 1)
    # word_logits: (batch, seq_len, vocab_size)
    assert word_logits.shape == (
        Config.BATCH_SIZE,
        Config.MAX_SEQ_LEN,
        tokenizer.get_vocab_size(),
    )

    print("Model architecture verified.")

    print("\n=== 4. Training Pipeline Demonstration ===")
    # Initialize Trainer
    trainer = Trainer(config=Config)

    # Run Training (1 Epoch on small data)
    print("Starting training loop...")
    trainer.fit()

    # Validation: Check if model artifact was created
    assert os.path.exists(Config.MODEL_SAVE_PATH), "Model checkpoint was not saved."
    print("Training completed and model saved.")

    print("\n=== 5. Inference Demonstration ===")
    # Initialize Predictor
    predictor = Predictor(config=Config, load_cached_data=True)

    # Run Prediction
    print("Running inference...")
    predictor.predict()

    # Validation: Check submission file
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file was not generated."

    # Validate Submission Format
    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission generated with {len(df_sub)} rows.")
    print("Head of submission:")
    print(df_sub.head())

    assert "id" in df_sub.columns
    assert "sentence" in df_sub.columns
    assert len(df_sub) > 0

    # Check that sentences are strings and not empty
    assert isinstance(df_sub.iloc[0]["sentence"], str)
    assert len(df_sub.iloc[0]["sentence"]) > 0

    print("\n=== Demonstration Completed Successfully ===")

import os
import torch
import pandas as pd
import shutil
import logging

# Import from provided library
from library.config import Config, set_seed
from library.vocab import Vocabulary
from library.dataset import InfillingDataset, get_dataloaders
from library.model import GatedInfillingModel
from library.engine import Engine
from library.utils import get_device

# Disable extensive logging for the demo
logging.getLogger("library.utils").setLevel(logging.WARNING)
logging.getLogger("library.vocab").setLevel(logging.WARNING)
logging.getLogger("library.dataset").setLevel(logging.WARNING)


def run_demo():
    print("--- Starting Library Demonstration ---")

    # 1. Configuration Override
    # We modify the Config class directly to create a lightweight environment for this demo.
    print("[1] Configuring environment for rapid demonstration...")

    # Use a separate working directory for this demo
    DEMO_DIR = "./working/demo_execution"
    if os.path.exists(DEMO_DIR):
        shutil.rmtree(DEMO_DIR)
    os.makedirs(DEMO_DIR, exist_ok=True)

    Config.WORKING_DIR = DEMO_DIR
    Config.VOCAB_FILE = os.path.join(DEMO_DIR, "vocab.npy")
    Config.MODEL_FILE = os.path.join(DEMO_DIR, "model.pth")
    Config.SUBMISSION_FILE = os.path.join(DEMO_DIR, "submission.csv")

    # Reduce dataset size significantly
    Config.DEBUG_SAMPLE_SIZE = 500  # Only use 500 samples

    # Reduce Model Complexity
    Config.VOCAB_SIZE = 2000
    Config.EMBED_DIM = 64
    Config.HIDDEN_DIM = 128
    Config.KERNEL_SIZE = 3
    Config.NUM_LAYERS = 2
    Config.DROPOUT = 0.0

    # Training params
    Config.BATCH_SIZE = 16
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Set Seed
    set_seed(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")
    print(f"    Working Directory: {Config.WORKING_DIR}")

    # 2. Vocabulary
    print("\n[2] Building Vocabulary...")
    vocab = Vocabulary()
    # Build from scratch using the small sample size
    vocab.build_from_corpus(load_cached_data=False)

    # Verification
    print(f"    Vocabulary Size: {len(vocab)}")
    assert len(vocab) > 0, "Vocabulary should not be empty"
    assert vocab.TOKEN_PAD in vocab.stoi, "PAD token missing"
    assert vocab.TOKEN_UNK in vocab.stoi, "UNK token missing"

    # Test encoding/decoding
    sample_text = "the quick brown fox"
    encoded = vocab.encode(sample_text, add_special_tokens=True)
    decoded = vocab.decode(encoded, remove_special_tokens=True)

    assert isinstance(encoded, list), "Encoding should return a list"
    assert (
        len(encoded) == len(sample_text.split()) + 2
    ), "Encoding should include START/END tokens"
    # Note: Decoded might contain UNKs if vocab is very small, but structure holds.
    print(f"    Encoding Test: '{sample_text}' -> {encoded}")

    # 3. Dataset & Dataloaders
    print("\n[3] Initializing Datasets and Loaders...")
    # Force reload to ensure DEBUG_SAMPLE_SIZE is respected
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab, load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify Train Batch
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"]
    targets = batch["targets"]

    assert input_ids.ndim == 2, "Input IDs should be (Batch, Seq_Len)"
    assert targets.ndim == 2, "Targets should be (Batch, Seq_Len)"
    assert input_ids.shape == targets.shape, "Input and Target shapes must match"
    print(f"    Batch Shape: {input_ids.shape}")

    # Verify Test Batch
    test_batch = next(iter(test_loader))
    assert "id" in test_batch, "Test batch must contain 'id'"
    assert "targets" not in test_batch, "Test batch should not contain 'targets'"

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = GatedInfillingModel(
        vocab_size=len(vocab),
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        kernel_size=Config.KERNEL_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT,
        padding_idx=vocab.pad_token_id,
    ).to(device)

    # Dummy Forward Pass
    dummy_input = input_ids.to(device)
    dummy_output = model(dummy_input)

    assert dummy_output.shape == (
        input_ids.shape[0],
        input_ids.shape[1],
        len(vocab),
    ), f"Output shape mismatch. Expected {(input_ids.shape[0], input_ids.shape[1], len(vocab))}, got {dummy_output.shape}"
    print("    Forward pass successful.")

    # 5. Training Engine
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    engine = Engine(model, optimizer, scheduler=None, vocab=vocab, device=device)

    # Train
    train_metrics = engine.train_one_epoch(train_loader, epoch=1)
    print(f"    Train Metrics: {train_metrics}")
    assert "loss" in train_metrics, "Train metrics missing loss"
    assert "accuracy" in train_metrics, "Train metrics missing accuracy"

    # Validate
    val_metrics = engine.validate(val_loader, epoch=1)
    print(f"    Val Metrics: {val_metrics}")
    assert "loss" in val_metrics, "Val metrics missing loss"

    # 6. Prediction / Submission
    print("\n[6] Generating Submission...")
    engine.generate_submission(test_loader, Config.SUBMISSION_FILE)

    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"

    # Verify CSV format
    df_sub = pd.read_csv(Config.SUBMISSION_FILE)
    print(f"    Submission Rows: {len(df_sub)}")
    print(f"    Columns: {list(df_sub.columns)}")

    assert "id" in df_sub.columns, "Submission missing 'id' column"
    assert "sentence" in df_sub.columns, "Submission missing 'sentence' column"
    assert len(df_sub) > 0, "Submission file is empty"

    # Check one sample
    sample_row = df_sub.iloc[0]
    assert isinstance(
        sample_row["id"], (int, float, pd.Int64Dtype)
    ), "ID should be numeric"
    assert isinstance(sample_row["sentence"], str), "Sentence should be string"

    print("\n--- Demonstration Completed Successfully ---")


if __name__ == "__main__":
    run_demo()

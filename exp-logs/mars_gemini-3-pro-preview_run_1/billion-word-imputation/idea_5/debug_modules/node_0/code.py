import sys
import os
import torch
import pandas as pd
import numpy as np
import logging

# Ensure library is in path
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.vocab import Vocabulary
from library.data import get_dataloaders
from library.model import GlobalLocalTransformer
from library.loss import MultiObjectiveGapLoss
from library.engine import Engine

# ==========================================
# 1. Configuration Overrides for Demo
# ==========================================
# We modify Config to run a lightweight, fast demonstration
print("--- Configuring Demo Environment ---")
Config.DEBUG = True
Config.DEBUG_SIZE = 2000  # Small subset for speed
Config.VOCAB_SIZE = 1000  # Small vocabulary
Config.MAX_LEN = 64  # Shorter sequence length
Config.EMBED_DIM = 64  # Small model dimension
Config.HIDDEN_DIM = 64
Config.NUM_LAYERS = 2  # Shallow network
Config.NUM_HEADS = 4
Config.BATCH_SIZE = 16
Config.EPOCHS = 1
Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

# Redirect outputs to a demo directory
Config.WORKING_DIR = "./working/demo_execution"
Config.VOCAB_PATH = os.path.join(Config.WORKING_DIR, "vocab.npy")
Config.TRAIN_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "train_tokens.parquet")
Config.VAL_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "val_tokens.parquet")
Config.TEST_TOKENS_PATH = os.path.join(Config.WORKING_DIR, "test_tokens.parquet")
Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

# Create the directory
Config.setup()

# Silence verbose logs for the demo
logging.getLogger("Vocabulary").setLevel(logging.WARNING)
logging.getLogger("data").setLevel(logging.WARNING)
logging.getLogger("model").setLevel(logging.WARNING)
logging.getLogger("engine").setLevel(logging.WARNING)

if __name__ == "__main__":
    # ==========================================
    # 2. Initialization
    # ==========================================
    print("--- Initializing ---")
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Device: {device}")

    # ==========================================
    # 3. Vocabulary
    # ==========================================
    print("--- Building Vocabulary ---")
    vocab = Vocabulary()
    # Force build from scratch (load_cached_data=False) to use the debug subset
    vocab.build(load_cached_data=False)

    # Verification
    assert len(vocab) > 0, "Vocabulary is empty"
    assert vocab.pad_token in vocab.stoi, "PAD token missing"
    assert vocab.gap_token in vocab.stoi, "GAP token missing"
    print(f"Vocabulary size: {len(vocab)}")

    # ==========================================
    # 4. Data Loading
    # ==========================================
    print("--- Loading Data ---")
    # Force re-computation to ensure debug subset is used
    train_loader, val_loader, test_loader = get_dataloaders(
        vocab=vocab,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=False,
    )

    # Verify Batch Structure
    print("Verifying batch structure...")
    batch = next(iter(train_loader))

    # Check keys
    required_keys = ["input_ids", "gap_mask", "target_loc", "target_id", "row_id"]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check shapes
    input_ids = batch["input_ids"]
    target_loc = batch["target_loc"]
    target_id = batch["target_id"]

    B, L = input_ids.shape
    assert B <= Config.BATCH_SIZE, "Batch size mismatch"
    assert L <= Config.MAX_LEN, "Sequence length exceeds max length"
    assert target_loc.shape == (B,), "Target location shape mismatch"
    assert target_id.shape == (B,), "Target ID shape mismatch"

    print("Batch verification passed.")

    # ==========================================
    # 5. Model Initialization
    # ==========================================
    print("--- Initializing Model ---")
    model = GlobalLocalTransformer().to(device)

    # Verify Forward Pass
    print("Verifying forward pass...")
    dummy_input = input_ids.to(device)
    loc_logits, id_logits, hidden_states = model(dummy_input)

    # Check output shapes
    # loc_logits: (B, L)
    assert loc_logits.shape == (B, L), f"Loc logits shape mismatch: {loc_logits.shape}"
    # id_logits: (B, L, V)
    assert id_logits.shape == (
        B,
        L,
        Config.VOCAB_SIZE,
    ), f"ID logits shape mismatch: {id_logits.shape}"
    # hidden_states: (B, L, H)
    assert hidden_states.shape == (
        B,
        L,
        Config.EMBED_DIM,
    ), f"Hidden states shape mismatch: {hidden_states.shape}"

    print("Model verification passed.")

    # ==========================================
    # 6. Training Loop
    # ==========================================
    print("--- Starting Training (1 Epoch) ---")
    criterion = MultiObjectiveGapLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    engine = Engine(
        model=model,
        device=device,
        vocab=vocab,
        optimizer=optimizer,
        criterion=criterion,
    )

    # Train
    train_metrics = engine.train_one_epoch(train_loader, epoch=1)
    print(f"Train Metrics: {train_metrics}")

    assert "train_loss" in train_metrics, "Train loss missing from metrics"
    assert not np.isnan(train_metrics["train_loss"]), "Train loss is NaN"

    # Evaluate
    print("--- Starting Evaluation ---")
    val_metrics = engine.evaluate(val_loader, split="val")
    print(f"Val Metrics: {val_metrics}")

    assert "val_levenshtein" in val_metrics, "Levenshtein metric missing"

    # ==========================================
    # 7. Inference / Submission
    # ==========================================
    print("--- Generating Submission ---")
    engine.predict_submission(test_loader, Config.SUBMISSION_PATH)

    # Verify Submission File
    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    df_sub = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission rows: {len(df_sub)}")

    # Check columns
    assert (
        "id" in df_sub.columns and "sentence" in df_sub.columns
    ), "Submission columns invalid"

    # Check row count (should be equal to debug size or total test size depending on how dataset handles debug)
    # Note: InterleavedDataset uses Config.DEBUG_SIZE for all splits if DEBUG is True.
    # We set DEBUG_SIZE=2000.
    assert len(df_sub) > 0, "Submission file is empty"

    print("\n=== Demo Execution Completed Successfully ===")

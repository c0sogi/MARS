import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import warnings
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import get_loaders, get_test_loader
from library.model import TweetModel
from library.loss import TweetLoss
from library.engine import train_fn, eval_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Sentiment Analysis Tweet Extraction Demo ===")

    # 1. Configure for Speed
    # Override Config parameters to ensure fast execution
    print("\n[1] Configuring environment for fast demonstration...")
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 4
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.N_FOLDS = 2  # We will only use fold 0

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Configuration updated: DEBUG=True, Batch Size=4, Sample Size=20")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions (Jaccard Metric)...")
    s1 = "good morning"
    s2 = "good morning"
    s3 = "good night"
    s4 = "bad afternoon"

    score_perfect = jaccard(s1, s2)
    score_partial = jaccard(s1, s3)
    score_none = jaccard(s1, s4)

    print(f"    Jaccard('{s1}', '{s2}') = {score_perfect}")
    print(f"    Jaccard('{s1}', '{s3}') = {score_partial}")

    if score_perfect != 1.0:
        raise AssertionError("Jaccard score for identical strings should be 1.0")
    if score_none != 0.0:
        raise AssertionError("Jaccard score for disjoint strings should be 0.0")
    print("    Jaccard utility verification passed.")

    # 3. Data Pipeline
    print("\n[3] Initializing Data Pipeline...")
    # Initialize Tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(Config.TOKENIZER_PATH)
        print(f"    Tokenizer loaded: {Config.TOKENIZER_PATH}")
    except Exception as e:
        print(f"    Failed to load tokenizer: {e}")
        # Fallback for offline environments if needed, but assuming installed packages work
        raise e

    # Get DataLoaders (Fold 0)
    print("    Generating DataLoaders for Fold 0...")
    train_loader, val_loader = get_loaders(
        fold=0, tokenizer=tokenizer, load_cached_data=False
    )

    # Inspect one batch
    batch = next(iter(train_loader))
    ids = batch["ids"]
    mask = batch["mask"]
    start_targets = batch["start_targets"]
    end_targets = batch["end_targets"]

    print(f"    Batch keys: {list(batch.keys())}")
    print(f"    Input IDs shape: {ids.shape}")
    print(f"    Start Targets shape: {start_targets.shape}")

    if ids.shape[0] != Config.TRAIN_BATCH_SIZE:
        raise AssertionError(
            f"Expected batch size {Config.TRAIN_BATCH_SIZE}, got {ids.shape[0]}"
        )
    if ids.shape[1] != Config.MAX_LEN:
        raise AssertionError(
            f"Expected sequence length {Config.MAX_LEN}, got {ids.shape[1]}"
        )
    print("    Data loading verification passed.")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    device = Config.DEVICE
    model = TweetModel()
    model.to(device)
    print(f"    Model moved to device: {device}")

    # Forward Pass Verification
    print("    Running dummy forward pass...")
    ids = ids.to(device)
    mask = mask.to(device)
    token_type_ids = batch["token_type_ids"].to(device)

    start_logits, end_logits = model(ids, mask, token_type_ids)

    print(f"    Start Logits shape: {start_logits.shape}")
    print(f"    End Logits shape: {end_logits.shape}")

    if start_logits.shape != (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN):
        raise AssertionError("Output logits shape mismatch.")
    print("    Model forward pass verification passed.")

    # 5. Loss Function
    print("\n[5] Calculating Loss...")
    criterion = TweetLoss()
    start_targets = start_targets.to(device)
    end_targets = end_targets.to(device)

    loss = criterion(start_logits, end_logits, start_targets, end_targets)
    print(f"    Calculated Loss: {loss.item():.4f}")

    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")
    print("    Loss calculation verification passed.")

    # 6. Training Loop Demo
    print("\n[6] Running Training Loop (1 Epoch on debug subset)...")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = int(len(train_loader) * Config.EPOCHS)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"    Epoch 1/1 - Average Train Loss: {avg_train_loss:.4f}")

    # 7. Evaluation Loop Demo
    print("\n[7] Running Evaluation Loop...")
    avg_val_loss, avg_jaccard = eval_fn(val_loader, model, device)
    print(f"    Validation Loss: {avg_val_loss:.4f}")
    print(f"    Validation Jaccard Score: {avg_jaccard:.4f}")

    if not (0.0 <= avg_jaccard <= 1.0):
        raise AssertionError("Jaccard score out of bounds [0, 1].")
    print("    Training and Evaluation loops execution passed.")

    # 8. Test Loader Demo
    print("\n[8] Verifying Test Loader...")
    test_loader = get_test_loader(tokenizer)
    test_batch = next(iter(test_loader))
    print(f"    Test Batch keys: {list(test_batch.keys())}")

    if "start_targets" in test_batch:
        raise AssertionError("Test loader should not contain targets.")
    print("    Test loader verification passed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

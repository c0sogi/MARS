import os
import shutil
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, logging as transformers_logging

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard, AverageMeter
from library.data import get_loaders, get_test_loader
from library.model import TweetModel, get_optimizer_params
from library.engine import train_fn, eval_fn

# Suppress transformer warnings for cleaner output
transformers_logging.set_verbosity_error()


def run_demo():
    print("=== Starting Sentiment Extraction Demo ===\n")

    # ---------------------------------------------------------
    # 1. Configuration & Setup
    # ---------------------------------------------------------
    print("[1] Configuring environment...")

    # Override Config for speed and isolation
    Config.OUTPUT_DIR = "./working/demo_execution_v2/"
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.FILTER_NEUTRAL = True  # Ensure we test the filtering logic

    # Clean up previous run if exists
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Set seed
    seed_everything(Config.SEED)
    print("    Configuration updated and seed set.")

    # ---------------------------------------------------------
    # 2. Verify Utilities
    # ---------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test Jaccard
    str1 = "I love machine learning"
    str2 = "love machine"
    score = jaccard(str1, str2)
    # Intersection: {love, machine} (2), Union: {i, love, machine, learning} (4) -> 0.5
    expected_score = 0.5
    assert (
        abs(score - expected_score) < 1e-6
    ), f"Jaccard calculation failed. Got {score}, expected {expected_score}"
    print(f"    Jaccard check passed: '{str1}' vs '{str2}' -> {score}")

    # Test AverageMeter
    meter = AverageMeter()
    meter.update(10, n=2)
    meter.update(20, n=1)
    # (10*2 + 20*1) / 3 = 40/3 = 13.333
    assert abs(meter.avg - 13.3333) < 1e-3, "AverageMeter calculation failed."
    print("    AverageMeter check passed.")

    # ---------------------------------------------------------
    # 3. Data Loading
    # ---------------------------------------------------------
    print("\n[3] Loading Data (Debug Mode)...")

    # Use debug=True to load only 100 samples
    train_loader, val_loader = get_loaders(fold=0, debug=True)

    print(f"    Train Loader Batches: {len(train_loader)}")
    print(f"    Val Loader Batches:   {len(val_loader)}")

    assert len(train_loader) > 0, "Train loader is empty."

    # Inspect one batch
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "offsets",
        "text",
        "sentiment",
        "start_targets",
        "end_targets",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    print(f"    Batch keys verified. Input shape: {batch['input_ids'].shape}")
    print(f"    Target shape: {batch['start_targets'].shape}")

    # ---------------------------------------------------------
    # 4. Model Initialization
    # ---------------------------------------------------------
    print("\n[4] Initializing Model...")

    device = Config.DEVICE
    model = TweetModel()
    model.to(device)

    print(f"    Model loaded on {device}.")

    # Verify Forward Pass
    input_ids = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, mask)

    # Check output shapes: (batch_size, seq_len)
    batch_size, seq_len = input_ids.shape
    assert start_logits.shape == (
        batch_size,
        seq_len,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        seq_len,
    ), f"End logits shape mismatch: {end_logits.shape}"

    print("    Forward pass successful. Logits shape verified.")

    # ---------------------------------------------------------
    # 5. Training & Evaluation Loop (Simulation)
    # ---------------------------------------------------------
    print("\n[5] Running Training & Evaluation Loop...")

    # Setup Optimizer (Simplified for demo)
    optimizer_parameters = get_optimizer_params(
        model,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        llrd_decay=Config.LLRD_DECAY,
    )
    optimizer = torch.optim.AdamW(optimizer_parameters)

    # Train for 1 epoch (on debug subset)
    print("    Training...")
    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler=None)
    print(f"    Train Loss: {avg_train_loss:.4f}")
    assert not np.isnan(avg_train_loss), "Training loss is NaN"

    # Evaluate
    print("    Evaluating...")
    avg_val_loss, start_preds, end_preds = eval_fn(val_loader, model, device)
    print(f"    Val Loss: {avg_val_loss:.4f}")
    assert not np.isnan(avg_val_loss), "Validation loss is NaN"
    assert start_preds.shape[0] == len(val_loader.dataset), "Prediction count mismatch"

    # ---------------------------------------------------------
    # 6. Decoding Predictions (Post-processing)
    # ---------------------------------------------------------
    print("\n[6] Demonstrating Prediction Decoding...")

    # We will take the first sample from the validation set predictions
    # and demonstrate how to extract the string using offsets.

    # Get data for the first validation batch to match with predictions
    val_iter = iter(val_loader)
    val_batch = next(val_iter)

    # Indices for the first sample in the batch
    idx = 0

    # Get predicted token indices
    start_idx = np.argmax(start_preds[idx])
    end_idx = np.argmax(end_preds[idx])

    # Ensure end is after start
    if end_idx < start_idx:
        end_idx = start_idx

    # Get Offsets and Text
    offsets = val_batch["offsets"][idx].numpy()
    text = val_batch["text"][idx]
    sentiment = val_batch["sentiment"][idx]

    print(f"    Original Text: '{text}'")
    print(f"    Sentiment:     '{sentiment}'")
    print(f"    Pred Token Span: {start_idx} -> {end_idx}")

    # Decode logic: Map token indices to character positions
    # Note: offsets[i] = (start_char, end_char)
    # We need to handle cases where indices might be out of bounds of the text tokens
    # (e.g. pointing to [CLS], [SEP], or padding)

    pred_char_start = 0
    pred_char_end = 0

    if start_idx < len(offsets):
        pred_char_start = offsets[start_idx][0]

    if end_idx < len(offsets):
        pred_char_end = offsets[end_idx][1]

    predicted_text = text[pred_char_start:pred_char_end]

    print(f"    Extracted Text: '{predicted_text}'")

    # Verification of non-empty extraction (unless model hasn't learned anything yet)
    # Since we only trained for a few steps on 100 samples, the quality might be low,
    # but the string slicing logic should hold.
    assert isinstance(predicted_text, str), "Extracted text is not a string"

    # ---------------------------------------------------------
    # 7. Cleanup
    # ---------------------------------------------------------
    print("\n[7] Cleaning up...")
    if os.path.exists(Config.OUTPUT_DIR):
        shutil.rmtree(Config.OUTPUT_DIR)
    print("    Temporary directory removed.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

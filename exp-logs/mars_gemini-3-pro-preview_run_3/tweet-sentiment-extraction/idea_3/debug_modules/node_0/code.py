import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import logging as transformers_logging

# Import provided library components
from library.config import Config
from library.utils import set_seed, jaccard, decode_span
from library.dataset import TweetDataset, get_tokenizer, prepare_data
from library.model import TweetModel
from library.engine import train_fn, eval_fn

# Suppress transformer warnings for cleaner output
transformers_logging.set_verbosity_error()

if __name__ == "__main__":
    print("--- Starting Demonstration of Sentiment Extraction Pipeline ---")

    # 1. Configuration and Setup
    # We override epochs to 1 and use a small batch size for the demo
    config = Config(debug=True, epochs=1)
    config.train_batch_size = 8
    config.valid_batch_size = 8

    # Set seed for reproducibility
    set_seed(config.seed)
    print(f"Configuration loaded. Device: {config.device}")

    # 2. Data Preparation
    # Load tokenizer
    tokenizer = get_tokenizer(config)

    # Load and process data (using cached if available, otherwise creates it)
    # We force load_cached_data=False to demonstrate the processing logic,
    # but in a real run, True is preferred.
    full_df = prepare_data(config, load_cached_data=False)

    # OPTIMIZATION: Slice data to a tiny subset for speed demonstration
    subset_size = 50
    mini_df = full_df.head(subset_size).copy()

    # Split into train/val manually for this demo (80/20 split)
    split_idx = int(0.8 * len(mini_df))
    train_df = mini_df.iloc[:split_idx].reset_index(drop=True)
    val_df = mini_df.iloc[split_idx:].reset_index(drop=True)

    print(f"\nData loaded and sliced for demo:")
    print(f"  Train samples: {len(train_df)}")
    print(f"  Val samples:   {len(val_df)}")

    # Create Datasets
    train_dataset = TweetDataset(train_df, tokenizer, config.max_len, is_test=False)
    val_dataset = TweetDataset(val_df, tokenizer, config.max_len, is_test=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for simple demo execution
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.valid_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    # 3. Model Initialization and Forward Pass Verification
    model = TweetModel(config)
    model.to(config.device)

    print("\nModel initialized successfully.")

    # Fetch one batch to verify shapes
    sample_batch = next(iter(train_loader))
    input_ids = sample_batch["input_ids"].to(config.device)
    attention_mask = sample_batch["attention_mask"].to(config.device)
    token_type_ids = sample_batch["token_type_ids"].to(config.device)

    # Run forward pass (no grad)
    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask, token_type_ids)

    # Verify output shapes: (batch_size, seq_len)
    assert (
        start_logits.shape == input_ids.shape
    ), f"Expected start_logits shape {input_ids.shape}, got {start_logits.shape}"
    assert (
        end_logits.shape == input_ids.shape
    ), f"Expected end_logits shape {input_ids.shape}, got {end_logits.shape}"

    print("Forward pass verification: Success (Output shapes correct).")

    # 4. Utility Logic Verification
    print("\nVerifying Utility Functions...")

    # Test Jaccard
    s1 = "sentiment extraction is fun"
    s2 = "sentiment extraction"
    score = jaccard(s1, s2)
    # Intersection: {sentiment, extraction} (2), Union: {sentiment, extraction, is, fun} (4) -> 0.5
    assert (
        abs(score - 0.5) < 1e-6
    ), f"Jaccard calculation failed. Expected 0.5, got {score}"
    assert jaccard("same", "same") == 1.0, "Jaccard identity failed"
    assert jaccard("abc", "def") == 0.0, "Jaccard disjoint failed"
    print("  Jaccard function: Verified.")

    # Test Decode Span
    # Create synthetic probabilities where start=1 and end=2 is the clear winner
    # Seq len = 5
    start_probs = np.array([0.1, 0.8, 0.05, 0.05, 0.0])
    end_probs = np.array([0.1, 0.1, 0.7, 0.1, 0.0])

    # Joint prob at (1, 2) = 0.8 * 0.7 = 0.56.
    # Invalid span (2, 1) would be 0.05 * 0.1 = 0.005 but masked out.
    best_start, best_end = decode_span(start_probs, end_probs)

    assert (
        best_start == 1 and best_end == 2
    ), f"Decode span failed. Expected (1, 2), got ({best_start}, {best_end})"
    print("  Decode Span function: Verified.")

    # 5. Training Loop Demonstration
    print("\nStarting Training Loop (1 Epoch on subset)...")

    # Setup Optimizer and Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

    # Run Train Function
    train_loss = train_fn(
        train_loader,
        model,
        optimizer,
        config.device,
        scheduler=None,  # Skipping scheduler for demo
        criterion=criterion,
        config=config,
    )

    print(f"  Training finished. Average Train Loss: {train_loss:.4f}")

    # 6. Evaluation Loop Demonstration
    print("Starting Evaluation Loop...")

    val_loss, val_jaccard = eval_fn(val_loader, model, config.device, criterion)

    print(f"  Evaluation finished.")
    print(f"  Valid Loss:    {val_loss:.4f}")
    print(f"  Valid Jaccard: {val_jaccard:.4f}")

    print("\n--- Demonstration Complete ---")

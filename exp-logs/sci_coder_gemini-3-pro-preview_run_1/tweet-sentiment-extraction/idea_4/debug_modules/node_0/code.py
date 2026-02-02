import os
import sys
import shutil
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, jaccard
from library.data import process_data, TweetDataset
from library.model import TweetModel
from library.engine import loss_fn, decode_prediction


def run_demo():
    print("=== Starting Library Usage Demonstration ===")

    # 1. Configuration Setup
    # We subclass Config to create a lightweight configuration for this demo
    class DemoConfig(Config):
        DEBUG_SAMPLE_SIZE = 20  # Very small subset for speed
        TRAIN_BATCH_SIZE = 4
        VALID_BATCH_SIZE = 4
        EPOCHS = 1
        WORKING_DIR = "./working/demo_run/"  # Isolated directory
        FILTER_NEUTRAL_TRAIN = False  # Keep neutrals to test neutral handling
        USE_AWP = False  # Disable AWP for simple forward pass check

    config = DemoConfig()

    # Ensure clean state
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
    os.makedirs(config.WORKING_DIR, exist_ok=True)

    seed_everything(config.SEED)
    print("Configuration initialized.")

    # 2. Data Processing
    print("\n--- Demonstrating Data Processing ---")

    # Load raw metadata
    df_train = pd.read_csv(config.TRAIN_META_PATH)
    # Take a small subset
    df_subset = df_train.head(config.DEBUG_SAMPLE_SIZE).copy()
    print(f"Loaded subset of {len(df_subset)} rows.")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    # Run process_data
    # This function handles tokenization, offset mapping, and target generation
    (
        input_ids,
        attention_mask,
        start_labels,
        end_labels,
        span_masks,
        offsets,
        meta_df,
    ) = process_data(df_subset, tokenizer, config, mode="train", load_cached_data=False)

    # Verification: Check shapes
    expected_shape = (config.DEBUG_SAMPLE_SIZE, config.MAX_LEN)
    assert (
        input_ids.shape == expected_shape
    ), f"Input IDs shape mismatch: {input_ids.shape}"
    assert (
        start_labels.shape == expected_shape
    ), f"Start labels shape mismatch: {start_labels.shape}"

    # Verification: Check Gaussian targets
    # For a row with selected_text, the start_labels should sum to approx 1.0 (probability distribution)
    # We find a row that is NOT empty string in selected_text
    valid_indices = [
        i for i, t in enumerate(meta_df["selected_text"]) if len(str(t).strip()) > 0
    ]
    if valid_indices:
        idx = valid_indices[0]
        sum_probs = np.sum(start_labels[idx])
        # It might be 0 if the selected text wasn't found (rare/edge case) or 1 if found
        # Given the dataset quality, we expect it to be close to 1.0 if found
        if sum_probs > 0.1:
            assert (
                0.99 <= sum_probs <= 1.01
            ), f"Gaussian target sum invalid: {sum_probs}"
            print(f"Gaussian target verification passed for index {idx}.")

    print("Data processing outputs verified.")

    # 3. Dataset and DataLoader
    print("\n--- Demonstrating Dataset & DataLoader ---")

    dataset = TweetDataset(
        input_ids,
        attention_mask,
        start_labels,
        end_labels,
        span_masks,
        offsets,
        meta_df["text"].values,
        meta_df["sentiment"].values,
        meta_df["selected_text"].values,
    )

    dataloader = DataLoader(dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=False)

    # Fetch one batch
    batch = next(iter(dataloader))

    # Verification: Check batch keys and tensor properties
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_labels",
        "end_labels",
        "span_masks",
        "raw_text",
        "sentiment",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key in batch: {key}"

    assert batch["input_ids"].shape == (config.TRAIN_BATCH_SIZE, config.MAX_LEN)
    print("DataLoader batch verification passed.")

    # 4. Model Initialization & Forward Pass
    print("\n--- Demonstrating Model Usage ---")

    device = config.DEVICE
    model = TweetModel(config)
    model.to(device)
    model.eval()  # Set to eval for deterministic check

    # Move batch to device
    b_input_ids = batch["input_ids"].to(device)
    b_mask = batch["attention_mask"].to(device)

    # Forward pass
    start_logits, end_logits, mask_logits = model(b_input_ids, b_mask)

    # Verification: Check output shapes
    # Logits should be (Batch, Max_Len)
    assert start_logits.shape == (
        config.TRAIN_BATCH_SIZE,
        config.MAX_LEN,
    ), "Start logits shape incorrect"
    assert end_logits.shape == (
        config.TRAIN_BATCH_SIZE,
        config.MAX_LEN,
    ), "End logits shape incorrect"
    if config.USE_AUX_HEAD:
        assert mask_logits.shape == (
            config.TRAIN_BATCH_SIZE,
            config.MAX_LEN,
        ), "Mask logits shape incorrect"

    print("Model forward pass verified.")

    # 5. Loss Calculation
    print("\n--- Demonstrating Loss Calculation ---")

    b_start_labels = batch["start_labels"].to(device)
    b_end_labels = batch["end_labels"].to(device)
    b_span_masks = batch["span_masks"].to(device)

    loss = loss_fn(
        start_logits,
        end_logits,
        mask_logits,
        b_start_labels,
        b_end_labels,
        b_span_masks,
        config,
    )

    # Verification: Loss validity
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss is negative"
    print(f"Calculated Loss: {loss.item():.4f}")

    # 6. Decoding Logic
    print("\n--- Demonstrating Decoding Logic ---")

    # Test Case A: Neutral Sentiment (Should return full text)
    text_a = "This is a neutral tweet."
    sentiment_a = "neutral"
    offsets_a = [(0, 0)] * config.MAX_LEN  # Dummy offsets
    # Logits don't matter for neutral
    pred_a = decode_prediction(
        torch.randn(config.MAX_LEN),
        torch.randn(config.MAX_LEN),
        text_a,
        offsets_a,
        sentiment_a,
    )
    assert pred_a == text_a, f"Neutral decoding failed. Got: '{pred_a}'"
    print("Neutral decoding logic verified.")

    # Test Case B: Positive/Negative Sentiment (Should extract span)
    text_b = "I love this library"
    sentiment_b = "positive"
    # Create fake offsets for "I"(0-1), "love"(2-6), "this"(7-11), "library"(12-19)
    # Simplified tokenization simulation
    offsets_b = np.zeros((config.MAX_LEN, 2), dtype=int)
    # Assume tokens: [CLS], I, love, this, library, [SEP]
    # Indices: 0, 1, 2, 3, 4, 5
    offsets_b[1] = [0, 1]
    offsets_b[2] = [2, 6]
    offsets_b[3] = [7, 11]
    offsets_b[4] = [12, 19]

    # Create logits that strongly favor "love" (token index 2)
    s_logits = torch.full((config.MAX_LEN,), -10.0)
    e_logits = torch.full((config.MAX_LEN,), -10.0)
    s_logits[2] = 10.0  # Start at "love"
    e_logits[2] = 10.0  # End at "love"

    pred_b = decode_prediction(s_logits, e_logits, text_b, offsets_b, sentiment_b)
    assert pred_b == "love", f"Span extraction failed. Expected 'love', got '{pred_b}'"
    print("Span extraction logic verified.")

    # 7. Cleanup
    print("\n--- Cleanup ---")
    if os.path.exists(config.WORKING_DIR):
        shutil.rmtree(config.WORKING_DIR)
        print(f"Removed temporary directory: {config.WORKING_DIR}")

    print("\n=== Demonstration Complete ===")


if __name__ == "__main__":
    run_demo()

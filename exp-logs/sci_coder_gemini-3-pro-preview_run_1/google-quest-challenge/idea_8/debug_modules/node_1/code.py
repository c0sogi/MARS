import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import warnings
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import from the provided library
from library.config import config
from library.utils import seed_everything, compute_spearman_metric
from library.data import get_dataloaders
from library.model import DebertaDualEncoder
from library.train import train_fn, eval_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def main():
    print("Starting Library Usage Demonstration...")

    # ---------------------------------------------------------
    # 1. Setup & Configuration Override
    # ---------------------------------------------------------
    # Override config for a fast debug run
    config.debug = True
    config.debug_sample_size = 64  # Small subset
    config.epochs = 1
    config.train_batch_size = 8
    config.valid_batch_size = 16
    config.working_dir = "./working/demo_run/"
    config.model_save_path = os.path.join(config.working_dir, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(config.working_dir, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(config.seed)
    device = torch.device(config.device)
    print(f"Device: {device}")
    print("Configuration configured for debug mode.")

    # ---------------------------------------------------------
    # 2. Data Pipeline Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Data Pipeline ---")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Get DataLoaders
    # Note: process_data will run (or load cache) and then get_dataloaders will slice for debug
    train_loader, val_loader, test_loader, meta_dims = get_dataloaders(
        config, tokenizer, load_cached_data=False  # Force process to ensure logic works
    )

    print(f"Meta Dimensions: {meta_dims}")

    # Verify Train Loader
    assert len(train_loader) > 0, "Train loader should not be empty."
    batch = next(iter(train_loader))

    # Check Batch Keys
    expected_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "category",
        "host",
        "labels",
    ]
    for key in expected_keys:
        assert key in batch, f"Batch missing key: {key}"

    # Check Shapes
    batch_size = batch["q_input_ids"].size(0)
    assert (
        batch_size == config.train_batch_size
    ), f"Expected batch size {config.train_batch_size}, got {batch_size}"
    assert (
        batch["labels"].size(1) == 30
    ), f"Expected 30 targets, got {batch['labels'].size(1)}"

    print("Data loading and batch structure verified.")

    # ---------------------------------------------------------
    # 3. Model Initialization & Forward Pass
    # ---------------------------------------------------------
    print("\n--- Verifying Model Architecture ---")

    model = DebertaDualEncoder(meta_dims)
    model.to(device)

    # Move batch to device
    q_input_ids = batch["q_input_ids"].to(device)
    q_attention_mask = batch["q_attention_mask"].to(device)
    a_input_ids = batch["a_input_ids"].to(device)
    a_attention_mask = batch["a_attention_mask"].to(device)
    category = batch["category"].to(device)
    host = batch["host"].to(device)

    # Forward Pass
    logits = model(
        q_input_ids, q_attention_mask, a_input_ids, a_attention_mask, category, host
    )

    # Verify Output Shape
    assert logits.shape == (
        batch_size,
        30,
    ), f"Output shape mismatch. Expected {(batch_size, 30)}, got {logits.shape}"
    print("Model forward pass successful. Output shape verified.")

    # ---------------------------------------------------------
    # 4. Metric Verification
    # ---------------------------------------------------------
    print("\n--- Verifying Metric Calculation ---")

    # Case 1: Perfect correlation
    y_true = np.random.rand(10, 30)
    y_pred = y_true.copy()
    score = compute_spearman_metric(y_true, y_pred)
    assert np.isclose(
        score, 1.0
    ), f"Metric should be 1.0 for identical inputs, got {score}"

    # Case 2: Random data (score should be valid float)
    y_pred_rand = np.random.rand(10, 30)
    score_rand = compute_spearman_metric(y_true, y_pred_rand)
    assert isinstance(score_rand, float), "Metric should return a float."
    assert -1.0 <= score_rand <= 1.0, "Metric should be between -1 and 1."

    print("Metric calculation verified.")

    # ---------------------------------------------------------
    # 5. Training Loop Simulation
    # ---------------------------------------------------------
    print("\n--- Verifying Training Loop ---")

    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=10
    )
    criterion = nn.BCEWithLogitsLoss()

    # Run training for one epoch (on debug subset)
    train_loss = train_fn(model, train_loader, optimizer, scheduler, criterion, device)

    assert not np.isnan(train_loss), "Training loss is NaN."
    print(f"Training loop complete. Loss: {train_loss:.4f}")

    # ---------------------------------------------------------
    # 6. Evaluation & Inference
    # ---------------------------------------------------------
    print("\n--- Verifying Evaluation & Inference ---")

    # Evaluation
    val_loss, val_score, val_preds = eval_fn(model, val_loader, criterion, device)
    assert not np.isnan(val_loss), "Validation loss is NaN."
    assert val_preds.shape[1] == 30, "Validation predictions have wrong shape."
    print(f"Validation complete. Score: {val_score:.4f}")

    # Inference (Test Set)
    test_preds = inference_fn(model, test_loader, device)

    # Check against test dataframe length (debug mode truncates test_df too)
    # We need to reload test_df to check expected length under debug mode
    test_df_path = os.path.join(config.working_dir, "test_processed.parquet")
    test_df = pd.read_parquet(test_df_path)
    if config.debug:
        test_df = test_df.iloc[: config.debug_sample_size]

    assert len(test_preds) == len(
        test_df
    ), f"Prediction count ({len(test_preds)}) mismatch with Test DF ({len(test_df)})"

    print("Inference complete. Prediction shapes verified.")

    print("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    main()

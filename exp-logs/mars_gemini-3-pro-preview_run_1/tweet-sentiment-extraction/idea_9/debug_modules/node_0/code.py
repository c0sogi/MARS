import os
import sys
import torch
import pandas as pd
import numpy as np
import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import SentimentDecoupledDeberta
from library.engine import run_training


def main():
    # 1. Setup and Configuration
    print("Initializing configuration...")

    # Suppress transformer warnings for cleaner output
    transformers.logging.set_verbosity_error()

    # Set seed for reproducibility
    seed_everything(Config.SEED)

    # Modify Config for a fast demonstration (Debug Mode)
    # We use a small subset and 1 epoch to ensure quick execution
    Config.DEBUG = True
    Config.DEBUG_SIZE = 100  # Small subset for demo
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 16

    # Ensure working directories exist (Config.setup() does this, but good to double check logic)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    print(f"Debug Mode: {Config.DEBUG}")
    print(f"Device: {Config.DEVICE}")

    # 2. Data Preparation
    print("\nPreparing DataLoaders...")
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=False,  # Force processing to demonstrate pipeline
        batch_size=Config.TRAIN_BATCH_SIZE,
        val_batch_size=Config.VALID_BATCH_SIZE,
        debug=Config.DEBUG,
        debug_size=Config.DEBUG_SIZE,
    )

    # Verification: Check Data Structure
    print("Verifying DataLoader output...")
    batch = next(iter(train_loader))

    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "text",
        "sentiment",
        "selected_text",
    ]
    for key in required_keys:
        assert key in batch, f"Missing key {key} in batch"

    # Check shapes
    input_ids = batch["input_ids"]
    assert input_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Incorrect input_ids shape: {input_ids.shape}"

    print("DataLoader verification passed.")

    # 3. Model Initialization
    print("\nInitializing Model...")
    model = SentimentDecoupledDeberta()
    model.to(Config.DEVICE)

    # Verification: Dummy Forward Pass
    print("Verifying Model Forward Pass...")
    with torch.no_grad():
        # Move batch to device
        b_input_ids = batch["input_ids"].to(Config.DEVICE)
        b_mask = batch["attention_mask"].to(Config.DEVICE)
        b_sentiment = batch["sentiment"]  # List of strings

        logits = model(b_input_ids, b_mask, b_sentiment)

        # Expected shape: (Batch_Size, Seq_Len, 2) -> 2 for start/end logits
        expected_shape = (Config.TRAIN_BATCH_SIZE, Config.MAX_LEN, 2)
        assert (
            logits.shape == expected_shape
        ), f"Incorrect output logits shape. Expected {expected_shape}, got {logits.shape}"

    print("Model verification passed.")

    # 4. Training Setup
    print("\nSetting up Optimizer and Scheduler...")
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # 5. Execution (Train -> Eval -> Save -> Predict)
    print("\nStarting Training Pipeline...")
    run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=Config.DEVICE,
        epochs=Config.EPOCHS,
    )

    # 6. Output Verification
    print("\nVerifying Outputs...")

    # Check Model Artifact
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"Model saved successfully at: {Config.MODEL_SAVE_PATH}")
        file_size = os.path.getsize(Config.MODEL_SAVE_PATH)
        assert file_size > 0, "Model file is empty"
    else:
        raise FileNotFoundError(f"Model file not found at {Config.MODEL_SAVE_PATH}")

    # Check Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        print(f"Submission file saved successfully at: {Config.SUBMISSION_PATH}")

        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission Shape: {sub_df.shape}")
        print("Submission Head:")
        print(sub_df.head())

        # Assertions
        assert (
            "textID" in sub_df.columns and "selected_text" in sub_df.columns
        ), "Submission missing required columns"
        assert (
            len(sub_df) == Config.DEBUG_SIZE
        ), f"Submission row count mismatch. Expected {Config.DEBUG_SIZE}, got {len(sub_df)}"

        # Check for non-empty predictions (simple check)
        # Note: It's possible for a model to predict empty strings, but unlikely for all
        non_empty_preds = sub_df["selected_text"].astype(str).str.strip().str.len() > 0
        if non_empty_preds.sum() == 0:
            print(
                "Warning: All predictions are empty. This might be expected in very early training or debug mode with random weights."
            )
        else:
            print("Submission contains valid text predictions.")

    else:
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    print("\nDemonstration completed successfully.")


if __name__ == "__main__":
    main()

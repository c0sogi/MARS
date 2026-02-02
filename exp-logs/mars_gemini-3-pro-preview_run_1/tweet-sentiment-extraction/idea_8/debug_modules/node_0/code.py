import os
import torch
import pandas as pd
import numpy as np
import shutil
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import load_and_cache_data
from library.model import MaskGuidedDeberta
from library.engine import train_fn, eval_fn, predict_test_set


def run_demo():
    print("=== Starting Demonstration of Tweet Sentiment Extraction Pipeline ===")

    # -------------------------------------------------------------------------
    # 1. Configuration Override
    # -------------------------------------------------------------------------
    # We modify the Config class directly to enable Debug mode and speed up execution.
    print("\n[1] Configuring environment for fast demonstration...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 samples for this demo
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8

    # Use a specific working directory for this demo to avoid conflicts with existing runs
    Config.WORKING_DIR = "./working/demo_execution"
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update paths that depend on WORKING_DIR
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.bin")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    # Ensure device is correctly set
    Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {Config.DEVICE}")
    print(f"Working Directory: {Config.WORKING_DIR}")

    # Set random seeds for reproducibility
    seed_everything(Config.SEED)

    # -------------------------------------------------------------------------
    # 2. Data Loading and Processing
    # -------------------------------------------------------------------------
    print("\n[2] Loading and Processing Data (Debug Mode)...")

    # We set load_cached_data=False to force reprocessing of the raw data
    # using the new DEBUG_SAMPLE_SIZE constraint.
    train_dataset = load_and_cache_data(subset="train", load_cached_data=False)
    val_dataset = load_and_cache_data(subset="val", load_cached_data=False)
    test_dataset = load_and_cache_data(subset="test", load_cached_data=False)

    print(f"Train Dataset Size: {len(train_dataset)}")
    print(f"Val Dataset Size: {len(val_dataset)}")
    print(f"Test Dataset Size: {len(test_dataset)}")

    # Verification
    assert len(train_dataset) > 0, "Train dataset should not be empty."
    # Note: Train dataset might be smaller than DEBUG_SAMPLE_SIZE due to neutral filtering
    assert (
        len(train_dataset) <= Config.DEBUG_SAMPLE_SIZE
    ), "Train dataset size exceeds debug limit."

    # Verify dataset item structure
    sample = train_dataset[0]
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "textID",
    ]
    for key in required_keys:
        assert key in sample, f"Dataset item missing key: {key}"

    assert (
        sample["input_ids"].shape[0] == Config.MAX_LEN
    ), "Input IDs have incorrect sequence length."

    # -------------------------------------------------------------------------
    # 3. Model Initialization
    # -------------------------------------------------------------------------
    print("\n[3] Initializing MaskGuidedDeberta Model...")
    model = MaskGuidedDeberta()
    model.to(Config.DEVICE)

    # Verify Model Components
    assert hasattr(model, "backbone"), "Model missing backbone."
    assert hasattr(model, "mask_head"), "Model missing mask_head."
    assert hasattr(model, "span_head"), "Model missing span_head."

    # Verify Forward Pass logic with a dummy batch
    print("Verifying forward pass...")
    dummy_loader = DataLoader(train_dataset, batch_size=2)
    dummy_batch = next(iter(dummy_loader))

    dummy_ids = dummy_batch["input_ids"].to(Config.DEVICE)
    dummy_mask = dummy_batch["attention_mask"].to(Config.DEVICE)

    with torch.no_grad():
        s_logits, e_logits, m_logits = model(dummy_ids, dummy_mask)

    # Check output shapes: (Batch, Seq_Len)
    expected_shape = (2, Config.MAX_LEN)
    assert (
        s_logits.shape == expected_shape
    ), f"Start logits shape mismatch: {s_logits.shape}"
    assert (
        e_logits.shape == expected_shape
    ), f"End logits shape mismatch: {e_logits.shape}"
    assert (
        m_logits.shape == expected_shape
    ), f"Mask logits shape mismatch: {m_logits.shape}"
    print("Forward pass successful.")

    # -------------------------------------------------------------------------
    # 4. Training Loop Execution
    # -------------------------------------------------------------------------
    print("\n[4] Running Training Loop (1 Epoch)...")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    num_training_steps = len(train_loader) * Config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    # Execute training
    train_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)

    print(f"Epoch 1 Train Loss: {train_loss:.4f}")
    assert not np.isnan(train_loss), "Training loss resulted in NaN."

    # -------------------------------------------------------------------------
    # 5. Evaluation Loop Execution
    # -------------------------------------------------------------------------
    print("\n[5] Running Evaluation Loop...")

    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    val_loss, val_jaccard = eval_fn(val_loader, model, Config.DEVICE)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Jaccard: {val_jaccard:.4f}")

    assert not np.isnan(val_loss), "Validation loss is NaN."
    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score is out of valid range [0, 1]."

    # -------------------------------------------------------------------------
    # 6. Inference and Submission Generation
    # -------------------------------------------------------------------------
    print("\n[6] Generating Predictions on Test Set...")

    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Generate predictions and save to CSV
    predict_test_set(test_loader, model, Config.DEVICE)

    # Verify Submission File
    assert os.path.exists(
        Config.SUBMISSION_PATH
    ), f"Submission file not found at {Config.SUBMISSION_PATH}"

    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Submission file created successfully with {len(submission_df)} rows.")
    print(submission_df.head(3))

    # Check format
    expected_columns = ["textID", "selected_text"]
    assert (
        list(submission_df.columns) == expected_columns
    ), f"Submission columns mismatch. Expected {expected_columns}"
    assert len(submission_df) == len(
        test_dataset
    ), "Submission row count does not match test dataset size."

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

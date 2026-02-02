import os
import sys
import shutil
import torch
import pandas as pd
import numpy as np
import warnings

# Import from the provided library
from library.config import TweetConfig
from library.utils import seed_everything
from library.data import get_loaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Tweet Sentiment Extraction Demo ===")

    # 1. Configuration Overrides for Speed and Demonstration
    # We modify the class attributes directly to affect all instances
    print("[Config] Setting up debug configuration...")
    TweetConfig.DEBUG = True
    TweetConfig.DEBUG_SIZE = 32  # Small subset for speed
    TweetConfig.TRAIN_BATCH_SIZE = 8
    TweetConfig.VALID_BATCH_SIZE = 8
    TweetConfig.EPOCHS = 1
    TweetConfig.NUM_WORKERS = 2

    # Setup temporary working directories
    base_work_dir = "./working/demo_run"
    TweetConfig.CACHE_DIR = os.path.join(base_work_dir, "cache")
    TweetConfig.SUBMISSION_DIR = os.path.join(base_work_dir, "submission")
    TweetConfig.SUBMISSION_PATH = os.path.join(
        TweetConfig.SUBMISSION_DIR, "submission.csv"
    )

    # Ensure directories exist
    os.makedirs(TweetConfig.CACHE_DIR, exist_ok=True)
    os.makedirs(TweetConfig.SUBMISSION_DIR, exist_ok=True)

    # Handle Sample Submission Mismatch for Debugging
    # generate_submission checks length against sample_submission.csv.
    # Since we are using DEBUG_SIZE, we need a matching sample submission file.
    print("[Setup] Creating temporary sample submission for debug size...")
    original_sample_sub = pd.read_csv("./input/sample_submission.csv")
    debug_sample_sub = original_sample_sub.head(TweetConfig.DEBUG_SIZE).copy()
    temp_sample_sub_path = os.path.join(base_work_dir, "sample_submission.csv")
    debug_sample_sub.to_csv(temp_sample_sub_path, index=False)

    # Override the path in config
    TweetConfig.SAMPLE_SUBMISSION_PATH = temp_sample_sub_path

    # Set seed for reproducibility
    seed_everything(TweetConfig.SEED)

    # 2. Data Loading
    print("\n[Data] Loading and processing data...")
    # force load_cached_data=False to demonstrate processing logic
    train_loader, val_loader, test_loader = get_loaders(load_cached_data=False)

    # Verify Data Loading
    print("[Data] Verifying data loaders...")
    try:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        assert input_ids.shape == (
            TweetConfig.TRAIN_BATCH_SIZE,
            TweetConfig.MAX_LEN,
        ), f"Expected input_ids shape {(TweetConfig.TRAIN_BATCH_SIZE, TweetConfig.MAX_LEN)}, got {input_ids.shape}"
        assert "start_tokens" in batch
        assert "sentiment" in batch
        print(f"    Batch verification passed. Input shape: {input_ids.shape}")
    except StopIteration:
        raise RuntimeError("Train loader is empty!")

    # 3. Model Initialization
    print("\n[Model] Initializing TweetModel...")
    device = TweetConfig.DEVICE
    # Using pretrained=False for speed and to avoid network dependency in this demo
    model = TweetModel(pretrained=False)
    model.to(device)

    # Verify Model Forward Pass
    print("[Model] Verifying forward pass...")
    with torch.no_grad():
        dummy_ids = batch["input_ids"].to(device)
        dummy_mask = batch["attention_mask"].to(device)
        s_logits, e_logits = model(dummy_ids, dummy_mask)
        assert s_logits.shape == (TweetConfig.TRAIN_BATCH_SIZE, TweetConfig.MAX_LEN)
        assert e_logits.shape == (TweetConfig.TRAIN_BATCH_SIZE, TweetConfig.MAX_LEN)
        print("    Forward pass verification passed.")

    # 4. Training Loop
    print("\n[Training] Starting training loop (1 Epoch)...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=TweetConfig.LEARNING_RATE)

    train_loss = train_fn(train_loader, model, optimizer, device)
    print(f"    Epoch 1/1 - Train Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"

    # 5. Evaluation
    print("\n[Evaluation] Running validation...")
    val_loss, val_jaccard = eval_fn(val_loader, model, device)
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation Jaccard: {val_jaccard:.4f}")

    # 6. Inference and Submission
    print("\n[Inference] Generating submission...")
    generate_submission(test_loader, model, device)

    # Verify Submission File
    if os.path.exists(TweetConfig.SUBMISSION_PATH):
        sub_df = pd.read_csv(TweetConfig.SUBMISSION_PATH)
        print(f"    Submission file created at {TweetConfig.SUBMISSION_PATH}")
        print(f"    Rows: {len(sub_df)}")
        print(f"    Columns: {list(sub_df.columns)}")

        # Check content
        assert len(sub_df) == TweetConfig.DEBUG_SIZE
        assert "selected_text" in sub_df.columns
        assert (
            not sub_df["selected_text"].isnull().any()
        ), "Submission contains null values"
    else:
        raise FileNotFoundError("Submission file was not created.")

    # 7. Cleanup
    print("\n[Cleanup] Removing temporary files...")
    if os.path.exists(base_work_dir):
        shutil.rmtree(base_work_dir)
    print("    Cleanup complete.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

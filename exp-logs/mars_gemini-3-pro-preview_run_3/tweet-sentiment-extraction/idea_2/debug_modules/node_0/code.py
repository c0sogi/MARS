import os
import pandas as pd
import torch
import torch.nn as nn
import transformers
import logging
import warnings
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, jaccard, get_selected_text
from library.dataset import get_data
from library.model import TweetModel
from library.loss import TweetLoss
from library.engine import train_fn, eval_fn

# Suppress warnings and logs for cleaner output
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)


def create_mini_datasets(num_samples=50):
    """
    Creates small subsets of the metadata CSVs to speed up the demo.
    """
    print(f"Creating mini datasets with {num_samples} samples each...")

    # Define paths
    mini_train_path = os.path.join(Config.WORKING_DIR, "mini_train.csv")
    mini_val_path = os.path.join(Config.WORKING_DIR, "mini_val.csv")
    mini_test_path = os.path.join(Config.WORKING_DIR, "mini_test.csv")

    # Load original metadata
    train_df = pd.read_csv(Config.TRAIN_META)
    val_df = pd.read_csv(Config.VAL_META)
    test_df = pd.read_csv(Config.TEST_META)

    # Sample and save
    train_df.head(num_samples).to_csv(mini_train_path, index=False)
    val_df.head(num_samples).to_csv(mini_val_path, index=False)
    test_df.head(num_samples).to_csv(mini_test_path, index=False)

    return mini_train_path, mini_val_path, mini_test_path


def test_utils():
    """
    Verifies utility functions.
    """
    print("\n--- Testing Utils ---")

    # Test Jaccard
    s1 = "hello world"
    s2 = "hello world"
    score = jaccard(s1, s2)
    assert score == 1.0, f"Expected Jaccard 1.0, got {score}"

    s1 = "hello world"
    s2 = "goodbye world"
    # Intersection: "world" (1), Union: "hello", "goodbye", "world" (3) -> 1/3
    score = jaccard(s1, s2)
    assert abs(score - 1 / 3) < 1e-6, f"Expected Jaccard 0.333..., got {score}"

    print("Utils verification passed.")


def run_demo():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Device: {device}")

    # 2. Optimize Config for Speed
    # Monkey-patch Config to use mini datasets and fewer epochs
    mini_train, mini_val, mini_test = create_mini_datasets(
        num_samples=32
    )  # Small batch size is 32, so 32 samples = 1 batch

    Config.TRAIN_META = mini_train
    Config.VAL_META = mini_val
    Config.TEST_META = mini_test
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo

    # Ensure cache directory is clean or distinct to avoid loading full cached data
    # We'll just disable loading cache in get_data call

    # 3. Data Loading
    print("\n--- Loading Data ---")
    # load_cached_data=False forces reprocessing of our new mini datasets
    train_loader, val_loader, test_loader = get_data(load_cached_data=False)

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    # Verify Batch Structure
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "text",
        "sentiment",
        "offsets",
    ]
    for key in required_keys:
        assert key in batch, f"Batch missing key: {key}"

    input_ids = batch["input_ids"]
    assert input_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Incorrect input_ids shape: {input_ids.shape}"

    print("Data loading verification passed.")

    # 4. Model Initialization
    print("\n--- Initializing Model ---")
    model = TweetModel()
    model.to(device)

    # Verify Forward Pass
    print("Running forward pass check...")
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    start_logits, end_logits = model(input_ids, attention_mask)

    assert start_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "Start logits shape mismatch"
    assert end_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "End logits shape mismatch"
    print("Model forward pass verification passed.")

    # 5. Loss Verification
    print("\n--- Verifying Loss ---")
    criterion = TweetLoss()
    start_pos = batch["start_positions"].to(device)
    end_pos = batch["end_positions"].to(device)

    loss = criterion(start_logits, end_logits, start_pos, end_pos)
    print(f"Calculated Loss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("Loss verification passed.")

    # 6. Training Integration (Engine)
    print("\n--- Running Training Loop (1 Epoch) ---")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    # Simple scheduler for demo
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.5, total_iters=1
    )

    # Run Train Function
    avg_train_loss = train_fn(train_loader, model, optimizer, device, scheduler)
    print(f"Average Train Loss: {avg_train_loss:.4f}")

    # Run Eval Function
    print("Running Evaluation...")
    avg_val_loss, avg_val_jaccard = eval_fn(val_loader, model, device)
    print(f"Average Val Loss: {avg_val_loss:.4f}")
    print(f"Average Val Jaccard: {avg_val_jaccard:.4f}")

    assert (
        avg_val_jaccard >= 0.0 and avg_val_jaccard <= 1.0
    ), "Jaccard score out of range"
    print("Engine verification passed.")

    # 7. Test get_selected_text logic manually with data from loader
    print("\n--- Verifying Post-Processing ---")
    # Take the first sample from the batch
    sample_text = batch["text"][0]
    sample_offsets = batch["offsets"][0].numpy()
    # Let's pretend the model predicted indices 1 to 3
    pred_start = 1
    pred_end = 3

    extracted = get_selected_text(sample_text, pred_start, pred_end, sample_offsets)
    print(f"Original: '{sample_text}'")
    print(f"Extracted (tokens {pred_start}-{pred_end}): '{extracted}'")

    assert isinstance(extracted, str), "Extracted text must be a string"
    print("Post-processing verification passed.")

    print("\nAll demonstrations completed successfully.")


if __name__ == "__main__":
    test_utils()
    run_demo()

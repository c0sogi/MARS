import os
import sys
import torch
import pandas as pd
import numpy as np
import tqdm


# 1. Silence TQDM to meet "Do not print progress bars" requirement
# We must patch it before importing library modules that use it.
def silent_tqdm(iterable=None, *args, **kwargs):
    if iterable is None:
        return SilentTqdmManual()
    return iterable


class SilentTqdmManual:
    def __init__(self):
        pass

    def update(self, n=1):
        pass

    def set_postfix(self, *args, **kwargs):
        pass

    def set_description(self, *args, **kwargs):
        pass

    def close(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass


tqdm.tqdm = silent_tqdm

# 2. Import Library Modules
from library import config
from library.config import Config
from library.data import get_train_val_loaders, get_test_dataloader
from library.model import TweetModel, run_idea_8
from library.utils import seed_everything

# 3. Patch Configuration for Speed
# We force debug mode and reduce folds/epochs to ensuring runtime < 1 hour (actually < 5 mins).
print(">>> Patching Config for fast demonstration...")

# Patch class attributes used by engine.py and model.py
Config.N_FOLDS = 2
Config.EPOCHS = 1
Config.TRAIN_BATCH_SIZE = 8
Config.VALID_BATCH_SIZE = 16
Config.DEBUG_SAMPLE_SIZE = 50  # Small sample for speed

# Patch __init__ to ensure Config().DEBUG is True in data.py
original_init = Config.__init__


def fast_init(self, debug=True, epochs=1, train_batch_size=8):
    # Force debug=True regardless of argument
    original_init(self, debug=True, epochs=epochs, train_batch_size=train_batch_size)


Config.__init__ = fast_init

# Ensure working directory is clean-ish or ready
if not os.path.exists(Config.WORKING_DIR):
    os.makedirs(Config.WORKING_DIR)

# 4. Main Execution Block
if __name__ == "__main__":
    seed_everything(Config.SEED)

    print("\n=== 1. Verifying Data Loading ===")
    # Load loaders for Fold 0
    # This triggers processing and caching.
    # Since we patched Config to debug=True, it should process only 50 rows.
    train_loader, val_loader = get_train_val_loaders(fold=0, load_cached_data=False)

    # Fetch a batch
    batch = next(iter(train_loader))

    # Verify Batch Keys
    expected_keys = {"input_ids", "attention_mask", "start_labels", "end_labels"}
    assert expected_keys.issubset(
        batch.keys()
    ), f"Batch missing keys. Found: {batch.keys()}"

    # Verify Shapes
    # Batch size should be Config.TRAIN_BATCH_SIZE (8)
    # Seq len should be Config.MAX_LEN (128)
    input_ids = batch["input_ids"]
    assert input_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Incorrect input_ids shape: {input_ids.shape}"

    print("Data Loading Verified: Batch shapes are correct.")

    print("\n=== 2. Verifying Model Architecture ===")
    device = Config.DEVICE
    model = TweetModel()
    model.to(device)
    model.eval()

    # Move batch to device
    b_input_ids = batch["input_ids"].to(device)
    b_att_mask = batch["attention_mask"].to(device)

    # Forward Pass
    with torch.no_grad():
        start_logits, end_logits = model(b_input_ids, b_att_mask)

    # Verify Output Shapes
    # Should be (Batch_Size, Seq_Len)
    assert start_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"End logits shape mismatch: {end_logits.shape}"

    print("Model Architecture Verified: Forward pass successful.")

    print("\n=== 3. Running Full Pipeline (Training + Inference) ===")
    # This runs the run_idea_8 function which orchestrates training and submission generation.
    # Due to our patches, it runs 1 Fold, 1 Epoch, on 50 samples.
    run_idea_8()

    print("\n=== 4. Verifying Submission ===")
    submission_path = Config.SUBMISSION_PATH

    if not os.path.exists(submission_path):
        raise FileNotFoundError(f"Submission file not found at {submission_path}")

    df_sub = pd.read_csv(submission_path)

    # Verify columns
    assert "textID" in df_sub.columns, "Submission missing textID column"
    assert "selected_text" in df_sub.columns, "Submission missing selected_text column"

    # Verify row count
    # In debug mode, test set is also subsampled to DEBUG_SAMPLE_SIZE (50)
    print(f"Submission Shape: {df_sub.shape}")
    assert len(df_sub) > 0, "Submission is empty"

    # Check for NaNs
    assert not df_sub.isnull().values.any(), "Submission contains NaN values"

    print("Submission Verified: Format and content look correct.")
    print("\n>>> Demonstration Complete.")

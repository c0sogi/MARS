import torch
import pandas as pd
import os
import shutil
import numpy as np
from library.config import Config
from library.utils import seed_everything, jaccard, loss_fn
from library.data import get_dataloaders
from library.model import TweetModel
from library.engine import train_fn, eval_fn, infer_fn


def run_demo():
    print("=== Starting Sentiment Extraction Library Demo ===")

    # 1. Configuration Setup for Rapid Execution
    # We override specific Config attributes to ensure the demo runs quickly
    # and uses a small subset of data.
    seed_everything(Config.SEED)

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 50  # Use only 50 rows for demonstration
    Config.EPOCHS = 1  # Train for only 1 epoch
    Config.TRAIN_BATCH_SIZE = 8
    Config.VALID_BATCH_SIZE = 8
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for this small demo

    # Ensure a clean state for data caching to verify processing logic
    if os.path.exists(Config.WORKING_DIR):
        shutil.rmtree(Config.WORKING_DIR)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(
        f"Configuration: Device={Config.DEVICE}, Debug={Config.DEBUG}, Batch Size={Config.TRAIN_BATCH_SIZE}"
    )

    # 2. Verify Utility Functions
    print("\n[1/6] Verifying Utility Functions...")
    # Test Jaccard similarity
    s1, s2 = "selected text", "selected text"
    s3 = "selected"
    assert jaccard(s1, s2) == 1.0, "Jaccard calculation failed for identical strings"
    assert 0.0 < jaccard(s1, s3) < 1.0, "Jaccard calculation failed for partial overlap"
    print(" - Jaccard metric: OK")

    # 3. Verify Data Loading
    print("\n[2/6] Verifying Data Loading...")
    # load_cached_data=False forces the pre-processing logic to execute
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    print(f" - Train Batches: {len(train_loader)}")
    print(f" - Val Batches:   {len(val_loader)}")
    print(f" - Test Batches:  {len(test_loader)}")

    # Inspect a single batch to ensure correctness
    batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "offsets",
        "text",
        "sentiment",
    ]
    for k in required_keys:
        assert k in batch, f"Batch missing key: {k}"

    # Check tensor shapes
    input_ids = batch["input_ids"]
    assert input_ids.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), f"Unexpected input_ids shape: {input_ids.shape}"
    print(" - Data Loader structure: OK")

    # 4. Verify Model Architecture
    print("\n[3/6] Verifying Model Architecture...")
    model = TweetModel()
    model.to(Config.DEVICE)

    # Perform a forward pass with the dummy batch
    b_input_ids = batch["input_ids"].to(Config.DEVICE)
    b_attention_mask = batch["attention_mask"].to(Config.DEVICE)

    start_logits, end_logits = model(b_input_ids, b_attention_mask)

    # Check output shapes: [batch_size, seq_len]
    assert start_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "Start logits shape mismatch"
    assert end_logits.shape == (
        Config.TRAIN_BATCH_SIZE,
        Config.MAX_LEN,
    ), "End logits shape mismatch"
    print(" - Model forward pass: OK")

    # 5. Verify Loss Calculation
    print("\n[4/6] Verifying Loss Calculation...")
    b_start_pos = batch["start_positions"].to(Config.DEVICE)
    b_end_pos = batch["end_positions"].to(Config.DEVICE)

    loss = loss_fn(start_logits, end_logits, b_start_pos, b_end_pos)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"
    print(f" - Computed Loss: {loss.item():.4f}")
    print(" - Loss function: OK")

    # 6. Verify Training, Evaluation, and Inference Loops
    print("\n[5/6] Running Training and Evaluation Loops...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Train for 1 epoch
    avg_train_loss = train_fn(train_loader, model, optimizer, Config.DEVICE)
    print(f" - Epoch 1 Training Loss: {avg_train_loss:.4f}")

    # Evaluate on validation set
    avg_val_loss, avg_val_jaccard = eval_fn(val_loader, model, Config.DEVICE)
    print(f" - Validation Loss: {avg_val_loss:.4f}")
    print(f" - Validation Jaccard: {avg_val_jaccard:.4f}")

    assert avg_train_loss > 0, "Training loss invalid"
    assert avg_val_loss > 0, "Validation loss invalid"
    print(" - Engine loops: OK")

    print("\n[6/6] Verifying Inference and Submission...")
    # Run inference on test set
    infer_fn(test_loader, model, Config.DEVICE)

    # Check if submission file exists and is valid
    if not os.path.exists(Config.SUBMISSION_PATH):
        raise FileNotFoundError(
            f"Submission file not found at {Config.SUBMISSION_PATH}"
        )

    submission = pd.read_csv(Config.SUBMISSION_PATH)
    print(f" - Submission file loaded. Shape: {submission.shape}")
    print(submission.head(3))

    assert list(submission.columns) == [
        "textID",
        "selected_text",
    ], "Submission columns mismatch"
    assert len(submission) > 0, "Submission file is empty"

    # Check for quoted strings in prediction (as per requirement, though logic handles raw strings,
    # the output CSV format handles quoting automatically. We check content validity).
    assert (
        not submission["selected_text"].isnull().any()
    ), "Submission contains null values"
    print(" - Inference generation: OK")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

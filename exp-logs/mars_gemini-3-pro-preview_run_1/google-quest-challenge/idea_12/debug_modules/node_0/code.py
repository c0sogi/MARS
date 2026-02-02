import os
import sys
import shutil
import torch
import numpy as np
import pandas as pd
import logging
import warnings
from transformers import AutoTokenizer

# Import library components
from library.config import Config
from library.utils import seed_everything, compute_spearman_metric
from library.dataset import get_dataloaders
from library.model import DistilRobertaDualEncoder
from library.engine import get_optimizer_params, train_one_epoch, validate, predict

# Suppress warnings and logs for clean output
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def run_demo():
    print("--- Starting Library Usage Demo ---")

    # 1. Setup Configuration for Fast Demo
    print("\n[1] Configuring environment...")

    # Set seed for reproducibility
    seed_everything(42)

    # Override Config for speed and isolation
    Config.DEBUG = True  # Limits data to 100 samples
    Config.NUM_EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.WORKING_DIR = "./working/demo_run"

    # Update cache paths to use the demo working directory
    Config.create_dirs()
    Config.TRAIN_CACHE = os.path.join(Config.WORKING_DIR, "train_cached.parquet")
    Config.VAL_CACHE = os.path.join(Config.WORKING_DIR, "val_cached.parquet")
    Config.TEST_CACHE = os.path.join(Config.WORKING_DIR, "test_cached.parquet")
    Config.MODEL_SAVE_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "submission.csv")

    print(f"    Working Directory: {Config.WORKING_DIR}")
    print(f"    Debug Mode: {Config.DEBUG}")

    # 2. Data Pipeline
    print("\n[2] Initializing Data Pipeline...")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Get DataLoaders (force reload to ensure debug slicing applies)
    train_loader, val_loader, test_loader = get_dataloaders(
        tokenizer, load_cached_data=False
    )

    print(f"    Train Batches: {len(train_loader)}")
    print(f"    Val Batches:   {len(val_loader)}")
    print(f"    Test Batches:  {len(test_loader)}")

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = [
        "q_input_ids",
        "q_attention_mask",
        "a_input_ids",
        "a_attention_mask",
        "labels",
    ]

    for key in required_keys:
        if key not in sample_batch:
            raise AssertionError(f"Missing key in batch: {key}")

    # Check shapes
    batch_size = sample_batch["q_input_ids"].size(0)
    num_targets = sample_batch["labels"].size(1)

    if batch_size != Config.TRAIN_BATCH_SIZE:
        raise AssertionError(
            f"Expected batch size {Config.TRAIN_BATCH_SIZE}, got {batch_size}"
        )
    if num_targets != 30:
        raise AssertionError(f"Expected 30 targets, got {num_targets}")

    print("    Batch structure verified successfully.")

    # 3. Model Initialization
    print("\n[3] Initializing Model...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")

    model = DistilRobertaDualEncoder()
    model.to(device)

    # Verify Forward Pass
    print("    Verifying forward pass...")
    with torch.no_grad():
        # Move sample batch to device
        inputs = {k: v.to(device) for k, v in sample_batch.items() if k != "labels"}
        logits = model(**inputs)

    if logits.shape != (batch_size, 30):
        raise AssertionError(
            f"Output shape mismatch. Expected {(batch_size, 30)}, got {logits.shape}"
        )

    print("    Forward pass successful. Output shape matches.")

    # 4. Metric Verification
    print("\n[4] Verifying Metric Calculation...")

    # Create synthetic data: Perfect correlation
    dummy_targets = np.random.rand(100, 30)
    dummy_preds = dummy_targets.copy()  # Perfect predictions

    score = compute_spearman_metric(dummy_preds, dummy_targets)
    print(f"    Perfect Correlation Score: {score:.4f}")

    if not np.isclose(score, 1.0):
        raise AssertionError("Metric calculation failed for perfect correlation.")

    # Random data
    dummy_preds_rand = np.random.rand(100, 30)
    score_rand = compute_spearman_metric(dummy_preds_rand, dummy_targets)
    print(f"    Random Correlation Score: {score_rand:.4f}")

    # 5. Training Loop Demo
    print("\n[5] Running Training Loop (1 Epoch)...")

    optimizer_params = get_optimizer_params(model)
    optimizer = torch.optim.AdamW(optimizer_params, lr=1e-4)

    # Train
    train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch=1)

    if not isinstance(train_loss, float) or train_loss < 0:
        raise AssertionError("Training loss is invalid.")

    # Validate
    print("    Running Validation...")
    val_loss, val_spearman = validate(model, val_loader, device)

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Spearman: {val_spearman:.4f}")

    # 6. Inference and Submission
    print("\n[6] Generating Predictions...")

    test_preds = predict(model, test_loader, device)

    if test_preds.shape[1] != 30:
        raise AssertionError("Prediction output has incorrect number of columns.")

    print(f"    Prediction shape: {test_preds.shape}")

    # Create submission dataframe
    # We need qa_ids from the test dataset.
    # Since we are in debug mode, we need to reload the test df processed by get_dataloaders logic
    # or just read the cache which was just created.
    test_df = pd.read_parquet(Config.TEST_CACHE)

    # Ensure lengths match (Debug mode slices data)
    if len(test_df) != len(test_preds):
        raise AssertionError(
            f"Mismatch between test dataframe ({len(test_df)}) and predictions ({len(test_preds)})"
        )

    submission = pd.DataFrame(test_preds, columns=Config.TARGET_COLS)
    submission.insert(0, "qa_id", test_df["qa_id"])

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"    Submission saved to {Config.SUBMISSION_PATH}")
    print(f"    Submission head:\n{submission.head(2)}")

    print("\n--- Demo completed successfully ---")


if __name__ == "__main__":
    run_demo()

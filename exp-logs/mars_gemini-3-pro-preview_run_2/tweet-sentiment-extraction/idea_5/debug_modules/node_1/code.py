import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_data, TweetDataset, TweetTestDataset
from library.model import TweetModel
from library.engine import train_fn, eval_fn, infer_fn


def run_demonstration():
    print("=== Starting Tweet Sentiment Extraction Library Demo ===")

    # ---------------------------------------------------------
    # 1. Configuration Setup for Fast Execution
    # ---------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")
    # Override Config settings to run on a tiny subset
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 60  # Small sample for speed
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 4
    Config.VALID_BATCH_SIZE = 8
    Config.N_FOLDS = 2  # Use fewer folds

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("    Debug Mode: Enabled")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # ---------------------------------------------------------
    # 2. Data Loading and Preprocessing
    # ---------------------------------------------------------
    print("\n[2] Loading and processing data...")
    # Force load_cached_data=False to ensure we generate the small debug dataset
    # instead of loading a potentially large full-dataset cache.
    train_data, test_data = get_data(load_cached_data=False)

    # Validate Data Structure
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "folds",
    ]
    for key in required_keys:
        if key not in train_data:
            raise AssertionError(f"Missing key '{key}' in processed train_data")

    print(f"    Train data keys: {list(train_data.keys())}")
    print(f"    Test data keys: {list(test_data.keys())}")
    print("    Data processing successful.")

    # ---------------------------------------------------------
    # 3. Dataset and DataLoader Instantiation
    # ---------------------------------------------------------
    print("\n[3] Creating Datasets and DataLoaders...")

    # Select Fold 0 for demonstration
    fold_idx = 0
    train_indices = np.where(train_data["folds"] != fold_idx)[0]
    val_indices = np.where(train_data["folds"] == fold_idx)[0]

    # Instantiate Datasets
    train_dataset = TweetDataset(train_data, indices=train_indices)
    val_dataset = TweetDataset(train_data, indices=val_indices)
    test_dataset = TweetTestDataset(test_data)

    # Instantiate DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 0 workers for simple debug execution
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.VALID_BATCH_SIZE, shuffle=False, num_workers=0
    )

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    print("    Fetched one batch from train_loader.")

    # Assert shapes
    batch_size = sample_batch["input_ids"].size(0)
    seq_len = sample_batch["input_ids"].size(1)

    assert (
        seq_len == Config.MAX_LEN
    ), f"Expected seq_len {Config.MAX_LEN}, got {seq_len}"
    assert (
        sample_batch["start_targets"].size(0) == batch_size
    ), "Target batch size mismatch"
    assert (
        sample_batch["attention_mask"].dtype == torch.long
    ), "Attention mask must be long tensor"

    print(f"    Batch Size: {batch_size}, Sequence Length: {seq_len}")
    print("    Dataset verification passed.")

    # ---------------------------------------------------------
    # 4. Model Initialization and Forward Pass
    # ---------------------------------------------------------
    print("\n[4] Initializing Model (DeBERTa-v3-large)...")
    model = TweetModel()
    model.to(Config.DEVICE)

    print("    Running forward pass check...")
    # Move sample batch to device
    inp_ids = sample_batch["input_ids"].to(Config.DEVICE)
    att_mask = sample_batch["attention_mask"].to(Config.DEVICE)

    # Forward pass
    start_logits, end_logits = model(inp_ids, att_mask)

    # Verify Output Shapes
    # Expected shape: (batch_size, seq_len)
    assert start_logits.shape == (
        batch_size,
        seq_len,
    ), f"Start logits shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        seq_len,
    ), f"End logits shape mismatch: {end_logits.shape}"

    print("    Forward pass successful. Logit shapes correct.")

    # ---------------------------------------------------------
    # 5. Training and Evaluation Loop
    # ---------------------------------------------------------
    print("\n[5] Running Training Loop (1 Epoch)...")

    # Optimizer setup
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler setup
    num_train_steps = int(len(train_dataset) / Config.TRAIN_BATCH_SIZE * Config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.WARMUP_RATIO),
        num_training_steps=num_train_steps,
    )

    # Run Train Function
    avg_train_loss = train_fn(train_loader, model, optimizer, Config.DEVICE, scheduler)
    print(f"    Training complete. Average Train Loss: {avg_train_loss:.4f}")

    # Assert Loss Validity
    if not (np.isfinite(avg_train_loss) and avg_train_loss > 0):
        raise AssertionError("Training loss is invalid (NaN, Inf, or <= 0)")

    print("\n[6] Running Evaluation Loop...")
    # Run Eval Function
    avg_val_loss, avg_jaccard = eval_fn(val_loader, model, Config.DEVICE)
    print(
        f"    Validation complete. Loss: {avg_val_loss:.4f}, Jaccard Score: {avg_jaccard:.4f}"
    )

    # Assert Metric Validity
    if not (0.0 <= avg_jaccard <= 1.0):
        raise AssertionError(f"Jaccard score {avg_jaccard} out of range [0, 1]")

    # ---------------------------------------------------------
    # 6. Inference and Submission Generation
    # ---------------------------------------------------------
    print("\n[7] Running Inference on Test Set...")
    predictions = infer_fn(test_loader, model, Config.DEVICE)

    print(f"    Inference complete. Generated {len(predictions)} predictions.")

    # Verify Prediction Format
    if len(predictions) > 0:
        first_pred = predictions[0]
        assert "textID" in first_pred, "Prediction missing 'textID'"
        assert "selected_text" in first_pred, "Prediction missing 'selected_text'"
        assert isinstance(
            first_pred["selected_text"], str
        ), "selected_text must be a string"
        print(f"    Sample Prediction: {first_pred}")

    # Create Submission DataFrame
    sub_df = pd.DataFrame(predictions)
    # Ensure columns are in correct order for submission
    sub_df = sub_df[["textID", "selected_text"]]

    # Save to working directory (simulated submission)
    output_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")
    sub_df.to_csv(output_path, index=False)
    print(f"    Submission file saved to: {output_path}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demonstration()

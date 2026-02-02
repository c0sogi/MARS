import os
import sys
import torch
import pandas as pd
import numpy as np
import warnings
import logging
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# =============================================================================
# 1. Environment Setup & Tqdm Patching
# =============================================================================
# Suppress warnings and transformers logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Monkeypatch tqdm to suppress progress bars (Requirement: "Do not print progress bars")
# This must be done BEFORE importing library modules that use tqdm.
import tqdm


def silent_tqdm(iterable, *args, **kwargs):
    """A silent pass-through for tqdm."""
    return iterable


tqdm.tqdm = silent_tqdm

# =============================================================================
# 2. Library Imports
# =============================================================================
from library.config import Config
from library.utils import seed_everything
from library.data import get_loaders
from library.model import SentimentModel
from library.engine import train_fn, eval_fn, predict


def main():
    print(">>> Starting Tweet Sentiment Extraction Demonstration")

    # =========================================================================
    # 3. Configuration Overrides for Fast Execution
    # =========================================================================
    # We override Config attributes to run a fast debug session
    Config.debug = True
    Config.debug_sample_size = 64  # Small subset for speed
    Config.epochs = 1
    Config.train_batch_size = 8
    Config.valid_batch_size = 16
    Config.num_workers = 0  # Disable multiprocessing for simple script execution

    # Define a specific model save path for this demo
    Config.model_save_path = os.path.join(Config.working_dir, "demo_best_model.bin")

    # Ensure working directory exists
    os.makedirs(Config.working_dir, exist_ok=True)

    # Set seeds for full reproducibility
    seed_everything(Config.seed)
    print(f"Configuration: Debug={Config.debug}, Device={Config.device}")

    # =========================================================================
    # 4. Data Loading & Verification
    # =========================================================================
    print("\n[1/4] Preparing Data...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Force reprocessing (load_cached_data=False) to ensure we use our small debug subset
    train_loader, val_loader, test_loader = get_loaders(
        tokenizer, load_cached_data=False
    )

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader should not be empty."
    assert len(val_loader) > 0, "Validation loader should not be empty."
    assert len(test_loader) > 0, "Test loader should not be empty."

    # Verify Batch Structure
    sample_batch = next(iter(train_loader))
    required_keys = [
        "input_ids",
        "attention_mask",
        "start_targets",
        "end_targets",
        "text",
        "sentiment",
    ]
    for key in required_keys:
        assert key in sample_batch, f"Batch missing key: {key}"

    print(f"Data loaded successfully. Train batches: {len(train_loader)}")

    # =========================================================================
    # 5. Model Initialization & Verification
    # =========================================================================
    print("\n[2/4] Initializing Model...")
    model = SentimentModel(Config)
    model.to(Config.device)

    # Verify Forward Pass with a dummy batch
    input_ids = sample_batch["input_ids"].to(Config.device)
    attention_mask = sample_batch["attention_mask"].to(Config.device)

    with torch.no_grad():
        start_logits, end_logits = model(input_ids, attention_mask)

    # Assert Output Shapes: (batch_size, max_len)
    batch_size = input_ids.size(0)
    assert start_logits.shape == (
        batch_size,
        Config.max_len,
    ), f"Shape mismatch: {start_logits.shape}"
    assert end_logits.shape == (
        batch_size,
        Config.max_len,
    ), f"Shape mismatch: {end_logits.shape}"

    print("Model initialized and forward pass verified.")

    # =========================================================================
    # 6. Training Loop Demonstration
    # =========================================================================
    print("\n[3/4] Running Training Loop (1 Epoch)...")

    # Setup Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    num_train_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_train_steps * Config.warmup_ratio),
        num_training_steps=num_train_steps,
    )

    # Run Training
    train_loss = train_fn(train_loader, model, optimizer, Config.device, scheduler)
    print(f"Training Loss: {train_loss:.5f}")
    assert not np.isnan(train_loss), "Training loss is NaN."

    # Run Evaluation
    val_jaccard = eval_fn(val_loader, model, Config.device)
    print(f"Validation Jaccard Score: {val_jaccard:.5f}")
    assert 0.0 <= val_jaccard <= 1.0, "Jaccard score is out of valid range [0, 1]."

    # Save Model (Simulating the checkpointing logic)
    torch.save(model.state_dict(), Config.model_save_path)
    assert os.path.exists(Config.model_save_path), "Model checkpoint was not saved."

    # =========================================================================
    # 7. Inference & Submission Generation
    # =========================================================================
    print("\n[4/4] Generating Predictions...")

    predict(test_loader, model, Config.device, Config.submission_path)

    # Verify Submission File
    assert os.path.exists(Config.submission_path), "Submission file not found."

    df_submission = pd.read_csv(Config.submission_path)
    print(f"Submission saved to: {Config.submission_path}")
    print(f"Submission Shape: {df_submission.shape}")
    print(f"First 3 rows:\n{df_submission.head(3)}")

    # Verify Columns
    assert "textID" in df_submission.columns, "Missing 'textID' column."
    assert "selected_text" in df_submission.columns, "Missing 'selected_text' column."

    # Verify Content Types (selected_text should be string)
    assert (
        df_submission["selected_text"].dtype == object
    ), "selected_text column should be object/string."

    print("\n>>> Demonstration Completed Successfully.")


if __name__ == "__main__":
    main()

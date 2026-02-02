import os
import sys
import torch
import pandas as pd
import numpy as np
import logging
import warnings
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import (
    get_data,
    PatentDataset,
    Collate,
    get_dataloaders,
    get_test_dataloader,
)
from library.model import PatentModel, get_optimizer_params
from library.engine import train_fn, valid_fn, inference_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_demo():
    print("=== Starting Demonstration of Library Components ===")

    # 1. Configuration Overrides for Speed
    # We modify the Config class attributes directly to run in a fast debug mode
    print("\n[1] Configuring environment...")
    Config.debug = True
    Config.debug_sample_size = 50  # Small subset for speed
    Config.epochs = 1
    Config.train_batch_size = 4
    Config.valid_batch_size = 8
    Config.num_workers = 0  # Avoid multiprocessing overhead for small demo

    # Ensure reproducibility
    seed_everything(Config.seed)
    logger = get_logger()
    logger.info(f"Debug Mode: {Config.debug}")
    logger.info(f"Device: {Config.device}")

    # 2. Data Loading and Processing
    print("\n[2] Loading and Processing Data...")
    # Force load from scratch to demonstrate processing logic
    train_df, val_df, test_df = get_data(load_cached_data=False)

    # Verification
    assert (
        len(train_df) == Config.debug_sample_size
    ), "Train DF size mismatch in debug mode"
    assert (
        "context_text" in train_df.columns
    ), "Context expansion failed (missing 'context_text')"
    print(f"Train DataFrame Shape: {train_df.shape}")
    print(f"Sample Context: {train_df.iloc[0]['context_text'][:50]}...")

    # 3. Tokenizer, Dataset, and DataLoader
    print("\n[3] Initializing Tokenizer and Datasets...")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    # Instantiate Dataset manually to check __getitem__
    ds = PatentDataset(train_df, tokenizer, Config.max_length)
    sample = ds[0]

    # Verification of Dataset output
    assert "input_ids" in sample
    assert "attention_mask" in sample
    assert "token_type_ids" in sample
    assert "label" in sample
    assert isinstance(sample["label"], float)
    print("Dataset __getitem__ check passed.")

    # Verification of Collate function
    collate_fn = Collate(tokenizer)
    batch_list = [ds[i] for i in range(Config.train_batch_size)]
    batch = collate_fn(batch_list)

    assert batch["input_ids"].shape[0] == Config.train_batch_size
    assert batch["input_ids"].shape[1] <= Config.max_length
    # Check padding (if lengths differed, padding token should be present)
    # In a small batch of 4, it's possible they are same length, but logic holds.
    print(f"Batch Input Shape: {batch['input_ids'].shape}")
    print("Collate function check passed.")

    # Get actual DataLoaders
    train_loader, val_loader = get_dataloaders(train_df, val_df, tokenizer)
    test_loader = get_test_dataloader(test_df, tokenizer)
    print(f"Train Loader Batches: {len(train_loader)}")

    # 4. Model Initialization
    print("\n[4] Initializing Model...")
    model = PatentModel(Config.model_name)
    model.to(Config.device)

    # Verify model structure
    assert hasattr(model, "model"), "PatentModel should wrap a HF model"
    print("Model initialized successfully.")

    # 5. Optimizer and LLRD (Layer-wise Learning Rate Decay)
    print("\n[5] Configuring Optimizer with LLRD...")
    optimizer_params = get_optimizer_params(
        model,
        base_lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        llrd_decay=Config.llrd_decay,
    )

    # Verification of LLRD
    # The first group (embeddings/bottom layers) should have a lower LR than the last group (head)
    # because decay < 1.0 and depth is higher for bottom layers.
    first_group_lr = optimizer_params[0]["lr"]
    last_group_lr = optimizer_params[-1]["lr"]

    print(f"LR for bottom layer (Embeddings): {first_group_lr:.2e}")
    print(f"LR for top layer (Head): {last_group_lr:.2e}")

    assert (
        first_group_lr < last_group_lr
    ), "LLRD failed: Embeddings LR should be lower than Head LR"

    optimizer = torch.optim.AdamW(optimizer_params)

    # Scheduler
    num_training_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(num_training_steps * Config.warmup_ratio),
        num_training_steps=num_training_steps,
    )

    # 6. Training Loop (Single Epoch)
    print("\n[6] Running Training Loop...")

    # We expect train_fn to return a float loss
    avg_train_loss = train_fn(
        model, train_loader, optimizer, scheduler, Config.device, epoch=0
    )

    print(f"Returned Train Loss: {avg_train_loss}")
    assert isinstance(avg_train_loss, float)
    assert avg_train_loss > 0, "Training loss should be positive"

    # 7. Validation Loop
    print("\n[7] Running Validation Loop...")
    avg_val_loss, metrics = valid_fn(model, val_loader, Config.device)

    print(f"Returned Val Loss: {avg_val_loss}")
    print(f"Returned Metrics: {metrics}")

    assert "pearson" in metrics
    # Pearson can be NaN if predictions are constant (std dev 0), but usually returns a float
    # With random init weights, it might be near 0.
    assert -1.0 <= metrics["pearson"] <= 1.0 or np.isnan(
        metrics["pearson"]
    ), "Pearson score out of range"

    # 8. Inference
    print("\n[8] Running Inference on Test Set...")
    predictions = inference_fn(model, test_loader, Config.device)

    print(f"Predictions Shape: {predictions.shape}")
    assert len(predictions) == len(
        test_df
    ), "Number of predictions must match test set size"
    assert predictions.dtype == np.float32 or predictions.dtype == np.float64

    # Check values are roughly in expected range (unbounded regression, but usually near 0-1 for this task)
    print(f"Prediction Sample: {predictions[:5]}")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

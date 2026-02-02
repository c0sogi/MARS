import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, Logger, get_score
from library.data import load_processed_data, InsultDataset, prepare_tapt_data
from library.model import InsultModel
from library.engine import get_optimizer_params, train_fn, valid_fn, inference_fn
from library.awp import AWP


def run_demo():
    print("Initializing Configuration...")
    # Initialize Config with debug=True
    config = Config(debug=True)

    # Overwrite model_name to a tiny model for rapid demonstration and logic verification
    config.model_name = "prajjwal1/bert-tiny"
    config.hidden_size = 128  # Size for bert-tiny
    config.train_batch_size = 4
    config.valid_batch_size = 4
    config.epochs = 1
    config.awp_start_epoch = 0  # Start AWP immediately to test it

    # Set seed
    seed_everything(config.seed)

    # Initialize Logger
    logger = Logger(os.path.join(config.output_dir, "demo_log.txt"))
    logger.log("Configuration loaded. Using model: " + config.model_name)

    # ---------------------------------------------------------
    # 1. Data Processing
    # ---------------------------------------------------------
    logger.log("\n--- Testing Data Loading & Processing ---")
    train_df, val_df, test_df = load_processed_data(config, load_cached_data=False)

    # Subset data for speed
    train_df = train_df.head(20).reset_index(drop=True)
    val_df = val_df.head(10).reset_index(drop=True)
    test_df = test_df.head(10).reset_index(drop=True)

    logger.log(f"Data loaded. Train shape: {train_df.shape}")

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create Datasets
    train_dataset = InsultDataset(train_df, tokenizer, config.max_length)
    val_dataset = InsultDataset(val_df, tokenizer, config.max_length)
    test_dataset = InsultDataset(test_df, tokenizer, config.max_length, is_test=True)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.valid_batch_size, shuffle=False, num_workers=0
    )

    # Verify DataLoader output
    batch = next(iter(train_loader))
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "target" in batch
    assert batch["input_ids"].shape == (config.train_batch_size, config.max_length)
    logger.log("DataLoader verification passed.")

    # ---------------------------------------------------------
    # 2. Model Initialization
    # ---------------------------------------------------------
    logger.log("\n--- Testing Model Initialization ---")
    device = config.device
    model = InsultModel(config, pretrained=True)
    model.to(device)

    # Verify Forward Pass
    with torch.no_grad():
        dummy_ids = batch["input_ids"].to(device)
        dummy_mask = batch["attention_mask"].to(device)
        logits = model(dummy_ids, dummy_mask)

    assert logits.shape == (
        config.train_batch_size,
        1,
    ), f"Expected shape {(config.train_batch_size, 1)}, got {logits.shape}"
    logger.log("Model forward pass verification passed.")

    # ---------------------------------------------------------
    # 3. Optimizer & Scheduler
    # ---------------------------------------------------------
    logger.log("\n--- Testing Optimizer & Scheduler ---")
    optimizer_parameters = get_optimizer_params(model, config)
    optimizer = torch.optim.AdamW(
        optimizer_parameters, lr=config.learning_rate, eps=1e-6
    )

    num_train_steps = len(train_loader) * config.epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    logger.log("Optimizer and Scheduler initialized.")

    # ---------------------------------------------------------
    # 4. AWP Initialization
    # ---------------------------------------------------------
    logger.log("\n--- Testing AWP Initialization ---")
    # AWP requires a scaler if used in mixed precision, but the class handles None gracefully until step
    # We will initialize it here. The actual scaler is created inside train_fn usually,
    # but for AWP class init we pass None or the scaler if we have it.
    # The provided train_fn creates a local scaler. We will pass scaler=None to init and let logic handle it.
    awp = AWP(
        model,
        optimizer,
        adv_lr=config.awp_lr,
        adv_eps=config.awp_eps,
        start_epoch=config.awp_start_epoch,
    )
    logger.log("AWP initialized.")

    # ---------------------------------------------------------
    # 5. Training Loop (Train Fn)
    # ---------------------------------------------------------
    logger.log("\n--- Testing Training Function (1 Epoch) ---")
    # We run the training function which handles the scaler, AWP attack steps, and backprop
    avg_loss = train_fn(
        train_loader,
        model,
        optimizer,
        device,
        scheduler,
        epoch=0,
        config=config,
        awp=awp,
    )

    assert not np.isnan(avg_loss), "Training loss returned NaN"
    logger.log(f"Training function executed successfully. Avg Loss: {avg_loss:.4f}")

    # ---------------------------------------------------------
    # 6. Validation Loop (Valid Fn)
    # ---------------------------------------------------------
    logger.log("\n--- Testing Validation Function ---")
    val_loss, val_preds = valid_fn(val_loader, model, device)

    assert len(val_preds) == len(val_df), "Validation predictions length mismatch"
    assert not np.isnan(val_loss), "Validation loss returned NaN"

    # Calculate dummy score
    val_score = get_score(val_df[config.target_col].values, val_preds)
    logger.log(f"Validation function executed successfully. AUC: {val_score:.4f}")

    # ---------------------------------------------------------
    # 7. Inference
    # ---------------------------------------------------------
    logger.log("\n--- Testing Inference Function ---")
    test_preds = inference_fn(test_loader, model, device)

    assert len(test_preds) == len(test_df), "Test predictions length mismatch"
    assert np.all(
        (test_preds >= 0) & (test_preds <= 1)
    ), "Predictions out of probability range [0,1]"
    logger.log("Inference function executed successfully.")

    # ---------------------------------------------------------
    # 8. TAPT Data Preparation
    # ---------------------------------------------------------
    logger.log("\n--- Testing TAPT Data Preparation ---")
    # This function aggregates text and saves to a file
    tapt_file = prepare_tapt_data(config, load_cached_data=False)

    assert os.path.exists(tapt_file), "TAPT corpus file was not created"
    with open(tapt_file, "r") as f:
        lines = f.readlines()

    logger.log(f"TAPT corpus generated at {tapt_file}. Contains {len(lines)} lines.")
    # Note: We skip run_tapt() as it involves a full training loop which takes too long for a demo.
    # The logic verification for training is covered by train_fn.

    logger.log("\nAll demonstrations and verifications passed successfully!")


if __name__ == "__main__":
    run_demo()

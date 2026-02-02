import os
import sys
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import InsultDataset, load_and_process_data
from library.model import InsultModel
from library.awp import AWP
from library.engine import train_fn, train_fn_awp, eval_fn

if __name__ == "__main__":
    print(">>> Starting Demo Script...")

    # 1. Configuration Setup
    # Initialize config and override for speed/demo purposes
    config = Config()
    config.debug = True
    config.debug_sample_size = 16  # Small subset for quick execution
    config.epochs_stage1 = 1
    config.epochs_stage2 = 1
    config.train_batch_size = 4
    config.gradient_accumulation_steps = 1
    config.awp_start_epoch = 0  # Start AWP immediately for demo

    # Use the first model in the list (roberta-large)
    model_name = config.model_names[0]

    # Set seed
    seed_everything(config.seed)
    logger = get_logger("demo")
    logger.info("Configuration initialized and optimized for speed.")

    # 2. Data Loading and Processing
    logger.info("Loading and processing data...")
    # Load small subset of training data
    train_df = load_and_process_data(
        "train", config=config, load_cached_data=False, debug=config.debug
    )

    # Verify data loading
    assert isinstance(train_df, pd.DataFrame), "Data should be a pandas DataFrame"
    assert (
        len(train_df) == config.debug_sample_size
    ), f"Expected {config.debug_sample_size} samples, got {len(train_df)}"
    assert (
        "Comment" in train_df.columns and "Insult" in train_df.columns
    ), "Missing required columns"

    # Initialize Tokenizer
    logger.info(f"Initializing tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create Dataset
    train_dataset = InsultDataset(
        texts=train_df["Comment"].values,
        tokenizer=tokenizer,
        max_len=config.max_len,
        targets=train_df["Insult"].values,
    )

    # Verify Dataset item
    sample_item = train_dataset[0]
    assert "input_ids" in sample_item, "Dataset item missing input_ids"
    assert "attention_mask" in sample_item, "Dataset item missing attention_mask"
    assert "target" in sample_item, "Dataset item missing target"
    assert (
        sample_item["input_ids"].shape[0] == config.max_len
    ), "Incorrect sequence length"

    # Create DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train_batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead for small demo
        pin_memory=True,
    )
    logger.info("Data pipeline ready.")

    # 3. Model Initialization
    logger.info("Initializing model...")
    device = config.device
    model = InsultModel(model_name, config=config, pretrained=True)
    model.to(device)

    # Verify Model Output Shape
    dummy_batch = next(iter(train_loader))
    dummy_ids = dummy_batch["input_ids"].to(device)
    dummy_mask = dummy_batch["attention_mask"].to(device)

    with torch.no_grad():
        dummy_out = model(dummy_ids, dummy_mask)

    assert dummy_out.shape == (
        config.train_batch_size,
        1,
    ), f"Expected output shape ({config.train_batch_size}, 1), got {dummy_out.shape}"
    logger.info("Model initialized and verified.")

    # 4. Optimizer and Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )

    num_train_steps = len(train_loader) * config.epochs_stage1
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    # 5. Training Loop Demonstration (Stage 1: Standard)
    logger.info("Running Stage 1 Training (Standard)...")
    avg_loss_stage1 = train_fn(
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epoch=0,
        config=config,
    )

    assert isinstance(avg_loss_stage1, float), "train_fn should return a float loss"
    assert not np.isnan(avg_loss_stage1), "Loss should not be NaN"
    logger.info(f"Stage 1 complete. Loss: {avg_loss_stage1:.4f}")

    # 6. AWP and Stage 2 Training Demonstration
    logger.info("Initializing AWP...")
    awp = AWP(model, optimizer, config)

    logger.info("Running Stage 2 Training (Adversarial)...")
    # Reset scheduler for stage 2 demo
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader)
    )

    avg_loss_stage2 = train_fn_awp(
        dataloader=train_loader,
        model=model,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        epoch=0,  # Ensure AWP triggers (awp_start_epoch=0)
        config=config,
        awp=awp,
    )

    assert isinstance(avg_loss_stage2, float), "train_fn_awp should return a float loss"
    assert not np.isnan(avg_loss_stage2), "Loss should not be NaN"
    logger.info(f"Stage 2 complete. Loss: {avg_loss_stage2:.4f}")

    # 7. Evaluation Demonstration
    logger.info("Running Evaluation...")
    # Use the same loader for eval demo
    eval_loss, preds = eval_fn(train_loader, model, device)

    assert isinstance(eval_loss, float), "eval_fn should return a float loss"
    assert isinstance(preds, np.ndarray), "Predictions should be a numpy array"
    assert len(preds) == len(
        train_dataset
    ), "Number of predictions must match dataset size"
    assert (preds >= 0).all() and (
        preds <= 1
    ).all(), "Predictions must be probabilities in [0, 1]"

    logger.info(
        f"Evaluation complete. Loss: {eval_loss:.4f}, Mean Pred: {preds.mean():.4f}"
    )

    print(">>> Demo Script Completed Successfully.")

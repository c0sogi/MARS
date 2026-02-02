import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

# Import from library
from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import load_and_process_data, ToxicityDataset
from library.model import DebertaV3MultiTask
from library.engine import train_one_epoch, validate, inference
from library.awp import AWP


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    # Override Config for Fast Baseline
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16  # Efficient for A100
    Config.ACCUMULATION_STEPS = 1
    Config.AWP_START_EPOCH = 0  # Apply AWP from the start

    seed_everything(Config.SEED)
    logger = get_logger()
    device = torch.device(Config.DEVICE)
    logger.info(f"Running on device: {device}")

    # ==========================================
    # 2. Data Loading & Processing
    # ==========================================
    logger.info("Loading and processing data...")

    # Load Training Data
    train_df = load_and_process_data(
        Config.TRAIN_PATH, "train_processed", is_train=True, debug=False
    )

    # Subsample Training Data for Fast Execution
    # Limit to 80,000 samples to ensure completion within 2 hours
    SAMPLE_SIZE = 80000
    if len(train_df) > SAMPLE_SIZE:
        logger.info(
            f"Subsampling training data from {len(train_df)} to {SAMPLE_SIZE}..."
        )
        train_df = train_df.sample(n=SAMPLE_SIZE, random_state=Config.SEED).reset_index(
            drop=True
        )

    # Load Validation Data (Full)
    val_df = load_and_process_data(
        Config.VAL_PATH, "val_processed", is_train=False, debug=False
    )

    # Initialize Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Create Datasets
    train_dataset = ToxicityDataset(train_df, tokenizer, Config.MAX_LEN)
    val_dataset = ToxicityDataset(val_df, tokenizer, Config.MAX_LEN)

    # Create Sampler for Training (Bias Mitigation)
    # Note: train_df has 'weight' column from load_and_process_data
    train_weights = torch.tensor(train_df["weight"].values, dtype=torch.double)
    train_sampler = WeightedRandomSampler(
        weights=train_weights, num_samples=len(train_weights), replacement=True
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        sampler=train_sampler,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # ==========================================
    # 3. Model Initialization
    # ==========================================
    logger.info(f"Initializing model: {Config.MODEL_NAME}")
    model = DebertaV3MultiTask(Config.MODEL_NAME)
    model.to(device)

    # ==========================================
    # 4. Optimizer & Scheduler
    # ==========================================
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_train_steps = int(len(train_loader) * Config.EPOCHS / Config.ACCUMULATION_STEPS)
    num_warmup_steps = int(num_train_steps * Config.WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    # Initialize AWP
    awp = AWP(
        model,
        optimizer,
        adv_lr=Config.AWP_LR,
        adv_eps=Config.AWP_EPS,
        start_epoch=Config.AWP_START_EPOCH,
    )

    # ==========================================
    # 5. Training Loop
    # ==========================================
    logger.info("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch, awp=awp
        )
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - Training Loss: {train_loss:.4f}"
        )

    # ==========================================
    # 6. Validation
    # ==========================================
    logger.info("Starting validation...")
    final_score, val_loss = validate(model, val_loader, device, val_df)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {final_score}")

    # ==========================================
    # 7. Failure Analysis
    # ==========================================
    logger.info("Performing Failure Analysis...")

    # Generate predictions on validation set for analysis
    model.eval()
    val_preds = []

    # Disable gradients for inference
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["ids"].to(device)
            mask = batch["mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)

            with torch.cuda.amp.autocast(enabled=Config.FP16):
                outputs = model(ids, mask, token_type_ids)

            probs = torch.sigmoid(outputs["toxicity_logits"]).view(-1).cpu().numpy()
            val_preds.append(probs)

    val_preds = np.concatenate(val_preds)

    # Calculate Error
    val_df["prediction"] = val_preds
    val_df["error"] = np.abs(val_df["target"] - val_df["prediction"])

    # Calculate Correlations
    print("-" * 40)
    print("Failure Analysis: Correlation with Error Magnitude")
    print("-" * 40)

    # Identity Correlations
    for col in Config.IDENTITY_COLS:
        if col in val_df.columns:
            # Fill NaNs with 0 for correlation calculation
            col_vals = val_df[col].fillna(0.0)
            corr = val_df["error"].corr(col_vals)
            print(f"Identity '{col}': {corr:.4f}")

    # Text Length Correlation
    val_df["text_len"] = val_df["comment_text"].astype(str).apply(len)
    len_corr = val_df["error"].corr(val_df["text_len"])
    print(f"Feature 'text_length': {len_corr:.4f}")
    print("-" * 40)

    # ==========================================
    # 8. Submission
    # ==========================================
    THRESHOLD = 0.9268315106992828

    if final_score > THRESHOLD:
        logger.info(
            f"Validation Score ({final_score:.6f}) exceeds threshold ({THRESHOLD:.6f}). Generating submission..."
        )

        # Load Test Data
        test_df = load_and_process_data(
            Config.TEST_PATH, "test_processed", is_train=False, debug=False
        )

        # Create Test Loader
        test_dataset = ToxicityDataset(test_df, tokenizer, Config.MAX_LEN, is_test=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=Config.PIN_MEMORY,
        )

        # Run Inference
        inference(
            model, test_loader, device, submission_path="./submission/submission.csv"
        )

    else:
        logger.info(
            f"Validation Score ({final_score:.6f}) did not exceed threshold ({THRESHOLD:.6f}). Skipping submission."
        )


if __name__ == "__main__":
    main()

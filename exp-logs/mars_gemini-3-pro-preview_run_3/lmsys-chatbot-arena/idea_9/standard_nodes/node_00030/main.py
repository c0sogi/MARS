import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler

# Import from provided library
from library.config import Config, seed_everything
from library.data import load_data, ChatbotDataset
from library.model import SiameseDebertaModel
from library.engine import train_one_epoch, validate, predict_with_tta
from library.utils import get_logger

# Setup Logger
logger = get_logger("runfile")

# Enforce Reproducibility
seed_everything(Config.seed)

# ==== Configuration Overrides for Fast Baseline ====
# Limit epochs and training data size to ensure execution within 2 hours
Config.epochs = 1
TRAIN_SUBSET_SIZE = 8000  # Sufficient for learning, small enough for speed


def perform_failure_analysis(model, dataloader, device):
    """
    Analyzes model failures on the validation set by correlating loss with input features.
    """
    logger.info("Running Failure Analysis...")
    model.eval()

    losses = []
    # Features are: [log(len_p), log(len_a), log(len_b)]
    feat_log_p = []
    feat_log_a = []
    feat_log_b = []

    # Use CrossEntropyLoss with reduction='none' to get per-sample loss
    criterion = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for batch in dataloader:
            input_ids_a = batch["input_ids_a"].to(device)
            attention_mask_a = batch["attention_mask_a"].to(device)
            input_ids_b = batch["input_ids_b"].to(device)
            attention_mask_b = batch["attention_mask_b"].to(device)
            features = batch["features"].to(device)
            targets = batch["target"].to(device)

            # Forward pass to get logits
            # Note: library model calculates loss internally if target is passed,
            # but we need raw logits to calculate per-sample loss manually.
            logits, _ = model(
                input_ids_a=input_ids_a,
                attention_mask_a=attention_mask_a,
                input_ids_b=input_ids_b,
                attention_mask_b=attention_mask_b,
                features=features,
                target=None,
            )

            # Calculate loss per sample
            batch_loss = criterion(logits, targets)

            # Collect data
            losses.extend(batch_loss.cpu().numpy())
            feat_log_p.extend(features[:, 0].cpu().numpy())
            feat_log_a.extend(features[:, 1].cpu().numpy())
            feat_log_b.extend(features[:, 2].cpu().numpy())

    # Convert to numpy arrays
    losses = np.array(losses)
    feat_log_p = np.array(feat_log_p)
    feat_log_a = np.array(feat_log_a)
    feat_log_b = np.array(feat_log_b)

    # Calculate Correlations
    # We check correlation between Error Magnitude (Loss) and Length features
    corr_p = np.corrcoef(losses, feat_log_p)[0, 1] if len(losses) > 1 else 0
    corr_a = np.corrcoef(losses, feat_log_a)[0, 1] if len(losses) > 1 else 0
    corr_b = np.corrcoef(losses, feat_log_b)[0, 1] if len(losses) > 1 else 0

    print(f"Correlation (Loss vs Log Prompt Len): {corr_p:.8f}")
    print(f"Correlation (Loss vs Log Resp A Len): {corr_a:.8f}")
    print(f"Correlation (Loss vs Log Resp B Len): {corr_b:.8f}")


def main():
    # 1. Load Data
    # We load cached data if available for speed
    train_df, val_df, test_df = load_data(load_cached_data=True)

    # Subset Training Data for Fast Baseline
    if len(train_df) > TRAIN_SUBSET_SIZE:
        logger.info(
            f"Subsetting training data from {len(train_df)} to {TRAIN_SUBSET_SIZE} samples."
        )
        train_df = train_df.iloc[:TRAIN_SUBSET_SIZE]

    # 2. Initialize Tokenizer and Datasets
    logger.info(f"Initializing Tokenizer: {Config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)

    logger.info("Creating Datasets...")
    train_dataset = ChatbotDataset(
        train_df, tokenizer, Config.max_length, is_test=False
    )
    # We must validate on the FULL validation set as per requirements
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.max_length, is_test=False)
    test_dataset = ChatbotDataset(test_df, tokenizer, Config.max_length, is_test=True)

    # 3. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.train_batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.valid_batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # 4. Initialize Model and Training Components
    device = Config.device
    logger.info(f"Initializing Model on {device}...")
    model = SiameseDebertaModel()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.learning_rate,
        weight_decay=Config.weight_decay,
        eps=Config.eps,
    )

    # Calculate steps for scheduler
    num_training_steps = (
        len(train_loader) * Config.epochs // Config.gradient_accumulation_steps
    )
    num_warmup_steps = int(num_training_steps * Config.warmup_ratio)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    scaler = GradScaler(enabled=Config.use_fp16)

    # 5. Training Loop
    logger.info("Starting Training...")
    for epoch in range(Config.epochs):
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, scaler, epoch
        )
        logger.info(f"Epoch {epoch+1} Train Loss: {train_loss:.6f}")

    # 6. Validation
    logger.info("Starting Validation...")
    val_loss_ce, val_log_loss = validate(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {val_log_loss}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission Logic
    threshold = 1.0005665522536111

    if val_log_loss < threshold:
        logger.info(
            f"Validation metric {val_log_loss} is better than threshold {threshold}. Generating submission..."
        )

        # Generate predictions with Test Time Augmentation
        predictions = predict_with_tta(model, test_loader, device)

        # Create Submission DataFrame
        # We read IDs from the metadata test file
        test_ids_df = pd.read_csv(Config.test_path)

        submission_df = pd.DataFrame(
            {
                "id": test_ids_df["id"],
                "winner_model_a": predictions[:, 0],
                "winner_model_b": predictions[:, 1],
                "winner_tie": predictions[:, 2],
            }
        )

        # Save
        submission_df.to_csv(Config.submission_path, index=False)
        logger.info(f"Submission saved to {Config.submission_path}")

    else:
        logger.info(
            f"Validation metric {val_log_loss} did not meet threshold {threshold}. Submission skipped."
        )


if __name__ == "__main__":
    main()

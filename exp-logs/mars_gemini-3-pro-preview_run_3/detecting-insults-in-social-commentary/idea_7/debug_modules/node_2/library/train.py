import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import load_processed_data, get_dataloader
from library.model import InsultModel
from library.awp import AWP

# Initialize Logger
logger = get_logger("train")


def train_fn(
    dataloader, model, criterion, optimizer, scheduler, awp, device, epoch, config
):
    """
    Training loop for one epoch with Gradient Accumulation and AWP.
    """
    model.train()

    # Use Mixed Precision for efficiency on A100
    scaler = torch.cuda.amp.GradScaler(enabled=True)

    total_loss = 0.0
    global_step = 0

    num_batches = len(dataloader)

    for step, batch in enumerate(dataloader):
        # Move inputs to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        targets = batch["target"].to(device).unsqueeze(1)  # (Batch, 1)

        batch_size = input_ids.size(0)

        # ==========================================
        # 1. Standard Forward Pass
        # ==========================================
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, targets)

        # Scale loss for gradient accumulation
        loss = loss / config.GRAD_ACCUM_STEPS

        # ==========================================
        # 2. Standard Backward Pass
        # ==========================================
        scaler.scale(loss).backward()

        # Accumulate loss for logging
        total_loss += loss.item() * config.GRAD_ACCUM_STEPS

        # ==========================================
        # 3. Optimization Step (with AWP)
        # ==========================================
        if (step + 1) % config.GRAD_ACCUM_STEPS == 0 or (step + 1) == num_batches:

            # Adversarial Weight Perturbation
            if config.USE_AWP and epoch >= config.AWP_START_EPOCH:
                # Perturb weights based on gradients from standard backward
                awp.attack()

                # Secondary Forward/Backward with perturbed weights
                with torch.cuda.amp.autocast(enabled=True):
                    adv_outputs = model(input_ids, attention_mask)
                    adv_loss = criterion(adv_outputs, targets)
                    # We don't divide adv_loss by grad_accum_steps here if we want strong regularization,
                    # but typically we match the scale. Let's match scale.
                    adv_loss = adv_loss / config.GRAD_ACCUM_STEPS

                scaler.scale(adv_loss).backward()

                # Restore original weights
                awp.restore()

            # Unscale gradients before clipping
            scaler.unscale_(optimizer)

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)

            # Optimizer Step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Scheduler Step
            if scheduler is not None:
                scheduler.step()

            global_step += 1

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def run_training(seed):
    """
    Runs the full training pipeline for a specific seed.
    """
    seed_everything(seed)
    logger.info(f"Starting training for Seed: {seed}")

    # ==========================================
    # 1. Data Loading & Preparation
    # ==========================================
    # Load Train and Validation Data
    logger.info("Loading datasets...")
    df_train = load_processed_data(Config.TRAIN_PATH, "train_data.parquet")
    df_val = load_processed_data(Config.VAL_PATH, "val_data.parquet")

    # Concatenate for Full Data Training (as per Idea)
    df_full = pd.concat([df_train, df_val]).reset_index(drop=True)

    if Config.DEBUG:
        df_full = df_full.iloc[: Config.DEBUG_SAMPLE_SIZE]
        logger.info(f"Debug Mode: Training on {len(df_full)} samples.")
    else:
        logger.info(f"Full Data Mode: Training on {len(df_full)} samples.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # DataLoader
    train_loader = get_dataloader(
        df_full,
        tokenizer,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        max_len=Config.MAX_LEN,
    )

    # ==========================================
    # 2. Model Initialization
    # ==========================================
    device = torch.device(Config.DEVICE)
    model = InsultModel(Config.MODEL_NAME)
    model.to(device)

    # ==========================================
    # 3. Optimizer & Scheduler
    # ==========================================
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Calculate total training steps
    num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
    max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
    num_warmup_steps = int(max_train_steps * Config.WARMUP_RATIO)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=max_train_steps
    )

    # ==========================================
    # 4. AWP Initialization
    # ==========================================
    awp = None
    if Config.USE_AWP:
        awp = AWP(
            model,
            optimizer,
            adv_lr=Config.AWP_LR,
            adv_eps=Config.AWP_EPS,
            start_epoch=Config.AWP_START_EPOCH,
        )
        logger.info("AWP Initialized.")

    # ==========================================
    # 5. Training Loop
    # ==========================================
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        avg_loss = train_fn(
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            awp,
            device,
            epoch,
            Config,
        )

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.6f} | Time: {elapsed:.0f}s"
        )

    # ==========================================
    # 6. Save Model
    # ==========================================
    save_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.bin")
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

    # Clear memory
    del model, optimizer, scheduler, train_loader, df_full
    torch.cuda.empty_cache()


def main():
    """
    Main function to execute training for all seeds.
    """
    logger.info("Initializing Training Pipeline...")

    for seed in Config.SEEDS:
        run_training(seed)

    logger.info("All training runs completed successfully.")


if __name__ == "__main__":
    main()

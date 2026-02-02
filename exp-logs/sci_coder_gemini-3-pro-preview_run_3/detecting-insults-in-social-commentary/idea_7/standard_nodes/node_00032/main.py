import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from scipy.stats import pearsonr

# Import library components
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import load_processed_data, get_dataloader
from library.model import InsultModel
from library.awp import AWP
from library.train import train_fn
from library.inference import inference_fn

# Initialize Logger
logger = get_logger("runfile")


def main():
    # ==========================================
    # 1. Configuration Overrides for Fast Baseline
    # ==========================================
    # We reduce epochs to ensure the run completes quickly while still learning.
    # We use 3 seeds to maintain the robustness of the ensemble strategy.
    Config.EPOCHS = 2
    Config.SEEDS = [42, 43, 44]
    Config.DEBUG = False

    logger.info("Configuration set for fast baseline.")
    logger.info(f"Epochs: {Config.EPOCHS}")
    logger.info(f"Seeds: {Config.SEEDS}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading datasets...")
    # Load Train and Validation sets separately to ensure strict hold-out validation
    df_train = load_processed_data(Config.TRAIN_PATH, "train_runfile.parquet")
    df_val = load_processed_data(Config.VAL_PATH, "val_runfile.parquet")

    logger.info(f"Train shape: {df_train.shape}")
    logger.info(f"Val shape: {df_val.shape}")

    # Initialize Tokenizer and Device
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
    device = torch.device(Config.DEVICE)

    # ==========================================
    # 3. Training Loop
    # ==========================================
    # We implement the loop here to control data splitting explicitly
    for seed in Config.SEEDS:
        seed_everything(seed)
        logger.info(f"--- Starting Training for Seed {seed} ---")

        # Prepare DataLoader for training
        train_loader = get_dataloader(
            df_train,
            tokenizer,
            batch_size=Config.TRAIN_BATCH_SIZE,
            shuffle=True,
            max_len=Config.MAX_LEN,
        )

        # Initialize Model
        model = InsultModel(Config.MODEL_NAME)
        model.to(device)

        # Initialize Optimizer
        optimizer = AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        num_update_steps_per_epoch = len(train_loader) // Config.GRAD_ACCUM_STEPS
        max_train_steps = Config.EPOCHS * num_update_steps_per_epoch
        num_warmup_steps = int(max_train_steps * Config.WARMUP_RATIO)

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        # Initialize AWP (Adversarial Weight Perturbation)
        awp = None
        if Config.USE_AWP:
            awp = AWP(
                model,
                optimizer,
                adv_lr=Config.AWP_LR,
                adv_eps=Config.AWP_EPS,
                start_epoch=Config.AWP_START_EPOCH,
            )

        # Loss Function
        criterion = nn.BCEWithLogitsLoss()

        # Train for specified epochs
        for epoch in range(Config.EPOCHS):
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
            logger.info(
                f"Seed {seed} | Epoch {epoch+1}/{Config.EPOCHS} | Loss: {avg_loss:.6f}"
            )

        # Save Model Weights
        save_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.bin")
        torch.save(model.state_dict(), save_path)
        logger.info(f"Model saved: {save_path}")

        # Cleanup to free memory
        del model, optimizer, scheduler, awp, train_loader
        torch.cuda.empty_cache()

    # ==========================================
    # 4. Validation & Failure Analysis
    # ==========================================
    logger.info("--- Starting Validation ---")

    # Prepare Validation DataLoader
    val_loader = get_dataloader(
        df_val,
        tokenizer,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        max_len=Config.MAX_LEN,
        is_test=True,  # We handle targets manually for metric calculation
    )

    # Ensemble Inference on Validation Set
    val_preds_accum = np.zeros(len(df_val))
    models_count = 0

    for seed in Config.SEEDS:
        model_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.bin")
        if not os.path.exists(model_path):
            continue

        model = InsultModel(Config.MODEL_NAME)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)

        # Run inference (no grad, eval mode handled by inference_fn)
        preds = inference_fn(model, val_loader, device)
        val_preds_accum += preds
        models_count += 1

        del model
        torch.cuda.empty_cache()

    # Average predictions
    avg_val_preds = val_preds_accum / models_count

    # Calculate Metric
    val_targets = df_val["Insult"].values
    auc_score = roc_auc_score(val_targets, avg_val_preds)

    print(f"Final Validation Metric: {auc_score}")

    # Failure Analysis
    logger.info("--- Failure Analysis ---")
    df_val["pred"] = avg_val_preds
    df_val["error"] = (df_val["Insult"] - df_val["pred"]).abs()

    # Calculate lengths (dataset is already decoded by load_processed_data)
    df_val["char_len"] = df_val["Comment"].apply(len)
    df_val["word_len"] = df_val["Comment"].apply(lambda x: len(str(x).split()))

    corr_char, _ = pearsonr(df_val["error"], df_val["char_len"])
    corr_word, _ = pearsonr(df_val["error"], df_val["word_len"])

    print(f"Correlation Error vs Char Length: {corr_char:.4f}")
    print(f"Correlation Error vs Word Length: {corr_word:.4f}")

    # ==========================================
    # 5. Submission
    # ==========================================
    threshold = 0.9639490968801314

    if auc_score > threshold:
        logger.info("Metric passed threshold. Generating submission...")

        # Load Test Data
        df_test = load_processed_data(
            Config.TEST_PATH, "test_runfile.parquet", debug=False
        )

        test_loader = get_dataloader(
            df_test,
            tokenizer,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            max_len=Config.MAX_LEN,
            is_test=True,
        )

        test_preds_accum = np.zeros(len(df_test))

        # Ensemble Inference on Test Set
        for seed in Config.SEEDS:
            model_path = os.path.join(Config.OUTPUT_DIR, f"model_seed_{seed}.bin")
            model = InsultModel(Config.MODEL_NAME)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)

            preds = inference_fn(model, test_loader, device)
            test_preds_accum += preds

            del model
            torch.cuda.empty_cache()

        avg_test_preds = test_preds_accum / len(Config.SEEDS)

        # Prepare Submission DataFrame
        if os.path.exists(Config.SAMPLE_SUBMISSION_PATH):
            sub_df = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)
        else:
            sub_df = df_test.copy()
            if "Insult" not in sub_df.columns:
                sub_df.insert(0, "Insult", 0.0)

        # Assign predictions
        sub_df["Insult"] = avg_test_preds

        # Save to file
        sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
        sub_df.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

    else:
        logger.info(
            f"Metric {auc_score} did not pass threshold {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()

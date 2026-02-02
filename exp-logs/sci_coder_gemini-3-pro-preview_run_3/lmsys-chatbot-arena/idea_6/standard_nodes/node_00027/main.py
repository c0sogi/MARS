import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data import ChatbotDataset, load_and_cache_data
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    logger = get_logger(name="runfile")
    device = Config.DEVICE

    # Configure for Full Run
    # Cite solution_lesson_node_00015: Data Volume Trumps Model Complexity.
    # We use the full dataset for 1 epoch.
    Config.EPOCHS = 1

    logger.info("Initializing Full Data Run...")

    # 2. Data Preparation
    tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

    # Load Training Data (Full)
    train_df = load_and_cache_data(Config.TRAIN_PATH, Config.TRAIN_CACHE_PATH)

    train_dataset = ChatbotDataset(
        train_df,
        tokenizer,
        Config.MAX_LENGTH,
        is_train=True,
        augment=Config.AUGMENT_DATA,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.TRAIN_BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    # Load Validation Data (Full)
    val_df = load_and_cache_data(Config.VAL_PATH, Config.VAL_CACHE_PATH)
    val_dataset = ChatbotDataset(val_df, tokenizer, Config.MAX_LENGTH, is_train=False)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.VALID_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Initialization
    model = SiameseDeberta()
    model.to(device)

    # 4. Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Steps calculation for scheduler
    num_update_steps_per_epoch = len(train_loader) // Config.ACCUMULATION_STEPS
    num_training_steps = num_update_steps_per_epoch * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 5. Training Loop
    logger.info("Starting Training...")
    for epoch in range(Config.EPOCHS):
        avg_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f"Epoch {epoch+1}/{Config.EPOCHS} - Avg Train Loss: {avg_loss:.6f}")

    # 6. Validation
    logger.info("Starting Validation...")
    val_results = eval_fn(model, val_loader, device)
    val_metric = val_results["metrics"]["log_loss"]

    # REQUIRED OUTPUT
    print(f"Final Validation Metric: {val_metric}")

    # 7. Failure Analysis
    logger.info("Performing Failure Analysis...")
    val_preds = val_results["predictions"]
    val_targets = val_df[["winner_model_a", "winner_model_b", "winner_tie"]].values

    # Clip predictions for numerical stability in log calculation
    val_preds = np.clip(val_preds, 1e-15, 1 - 1e-15)

    # Calculate Log Loss per sample
    # Log Loss = - sum(y_true * log(y_pred))
    sample_losses = -np.sum(val_targets * np.log(val_preds), axis=1)

    # Compute lengths
    val_df["len_prompt"] = val_df["prompt"].fillna("").astype(str).str.len()
    val_df["len_resp_a"] = val_df["response_a"].fillna("").astype(str).str.len()
    val_df["len_resp_b"] = val_df["response_b"].fillna("").astype(str).str.len()
    val_df["error_magnitude"] = sample_losses

    # Correlations
    corr_prompt = val_df["error_magnitude"].corr(val_df["len_prompt"])
    corr_resp_a = val_df["error_magnitude"].corr(val_df["len_resp_a"])
    corr_resp_b = val_df["error_magnitude"].corr(val_df["len_resp_b"])

    print("Failure Analysis - Error Magnitude Correlations:")
    print(f"Correlation with Prompt Length: {corr_prompt:.4f}")
    print(f"Correlation with Response A Length: {corr_resp_a:.4f}")
    print(f"Correlation with Response B Length: {corr_resp_b:.4f}")

    # 8. Submission
    THRESHOLD = 1.0061561136439758

    if val_metric < THRESHOLD:
        logger.info(
            f"Validation metric {val_metric} is better than threshold {THRESHOLD}. Generating submission..."
        )

        # Load Test Data
        test_df = load_and_cache_data(Config.TEST_PATH, Config.TEST_CACHE_PATH)
        test_dataset = ChatbotDataset(
            test_df, tokenizer, Config.MAX_LENGTH, is_train=False
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Inference
        test_results = eval_fn(model, test_loader, device)
        test_preds = test_results["predictions"]

        # Create Submission DataFrame
        submission = pd.DataFrame(
            {
                "id": test_df["id"],
                "winner_model_a": test_preds[:, 0],
                "winner_model_b": test_preds[:, 1],
                "winner_tie": test_preds[:, 2],
            }
        )

        # Save
        submission.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric {val_metric} did not meet threshold {THRESHOLD}. Submission skipped."
        )


if __name__ == "__main__":
    main()

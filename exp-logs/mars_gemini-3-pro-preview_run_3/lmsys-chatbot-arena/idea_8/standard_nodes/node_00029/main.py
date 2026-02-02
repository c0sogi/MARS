import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

# Import library modules
from library.config import Config
from library.data import get_dataloaders, load_dataset_df, ChatbotDataset, CollateFn
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn, inference_fn
from library.utils import seed_everything, get_logger


def main():
    # 1. Setup and Configuration Overrides
    # Override Config for a fast but effective baseline within 2 hours
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 16  # A100 can handle this size easily
    Config.VALID_BATCH_SIZE = 32

    # Initialize Logger
    logger = get_logger("main_script")
    seed_everything(Config.SEED)

    logger.info("Starting runfile.py execution...")
    logger.info(
        f"Configuration: Epochs={Config.EPOCHS}, Batch Size={Config.TRAIN_BATCH_SIZE}"
    )

    # 2. Data Loading
    logger.info("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    logger.info("Initializing model...")
    device = Config.DEVICE
    model = SiameseDeberta()
    model.to(device)

    # Optimizer and Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # 4. Training Loop
    best_val_loss = float("inf")

    for epoch in range(Config.EPOCHS):
        logger.info(f"Starting Epoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss = train_fn(model, train_loader, optimizer, scheduler, device, epoch)
        logger.info(f"Epoch {epoch + 1} Training Loss: {train_loss:.6f}")

        # Validate
        val_loss = eval_fn(model, val_loader, device)
        logger.info(f"Epoch {epoch + 1} Validation Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            logger.info(f"New best model found. Saving to {Config.MODEL_SAVE_PATH}")
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)

    # 5. Final Evaluation and Metric Reporting
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))
    model.to(device)

    final_val_loss = eval_fn(model, val_loader, device)

    # REQUIRED FORMAT: Print the final validation metric
    print(f"Final Validation Metric: {final_val_loss}")

    # 6. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Get predictions on validation set
    val_preds = inference_fn(model, val_loader, device)

    # Load validation metadata to get targets and text lengths
    val_df = load_dataset_df("val", load_cached_data=True)

    # Extract targets
    target_cols = ["winner_model_a", "winner_model_b", "winner_tie"]
    val_targets = val_df[target_cols].values

    # Calculate Cross Entropy per sample (Error Magnitude)
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    val_preds_clipped = np.clip(val_preds, epsilon, 1 - epsilon)
    # Cross Entropy = - sum(y_true * log(y_pred))
    error_magnitude = -np.sum(val_targets * np.log(val_preds_clipped), axis=1)

    # Calculate Input Features (Lengths)
    # We use simple string length or word count approximation
    val_df["len_prompt"] = val_df["prompt"].fillna("").str.len()
    val_df["len_res_a"] = val_df["response_a"].fillna("").str.len()
    val_df["len_res_b"] = val_df["response_b"].fillna("").str.len()
    val_df["diff_len"] = np.abs(val_df["len_res_a"] - val_df["len_res_b"])

    # Calculate Correlations
    corr_prompt = np.corrcoef(error_magnitude, val_df["len_prompt"])[0, 1]
    corr_res_a = np.corrcoef(error_magnitude, val_df["len_res_a"])[0, 1]
    corr_res_b = np.corrcoef(error_magnitude, val_df["len_res_b"])[0, 1]
    corr_diff = np.corrcoef(error_magnitude, val_df["diff_len"])[0, 1]

    print("\nFailure Analysis - Correlation with Error Magnitude:")
    print(f"Prompt Length: {corr_prompt:.4f}")
    print(f"Response A Length: {corr_res_a:.4f}")
    print(f"Response B Length: {corr_res_b:.4f}")
    print(f"Abs Length Diff: {corr_diff:.4f}")

    # 7. Submission Generation (Conditional)
    THRESHOLD = 1.0061561136439758
    if final_val_loss < THRESHOLD:
        logger.info("Validation metric meets threshold. Generating submission...")

        # Load Test Data
        test_df = load_dataset_df("test", load_cached_data=True)

        # --- TTA Step 1: Original Prediction ---
        logger.info("Predicting on original test set...")
        preds_orig = inference_fn(model, test_loader, device)

        # --- TTA Step 2: Swapped Prediction ---
        logger.info("Predicting on swapped test set (TTA)...")
        test_df_swapped = test_df.copy()
        # Swap columns
        test_df_swapped = test_df_swapped.rename(
            columns={"response_a": "response_b_temp", "response_b": "response_a_temp"}
        )
        test_df_swapped = test_df_swapped.rename(
            columns={"response_b_temp": "response_b", "response_a_temp": "response_a"}
        )

        # Create temporary dataset/loader for swapped data
        # We need the tokenizer from the original dataset to be safe, or re-init
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)

        swapped_dataset = ChatbotDataset(
            test_df_swapped, tokenizer, Config.MAX_LEN, is_test=True
        )
        collate_fn = CollateFn(tokenizer)
        swapped_loader = torch.utils.data.DataLoader(
            swapped_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        preds_swapped = inference_fn(model, swapped_loader, device)

        # --- TTA Step 3: Averaging ---
        # Model Output Classes: [0: A wins, 1: B wins, 2: Tie]
        # For swapped input (B, A):
        # Index 0 means "First input wins" -> B wins
        # Index 1 means "Second input wins" -> A wins
        # Index 2 means Tie

        final_preds = np.zeros_like(preds_orig)

        # Prob(A wins) = 0.5 * (Orig_A + Swapped_B_wins_which_is_index_1) -> Wait.
        # Swapped input is (B, A).
        # Output 0 -> First (B) wins. Output 1 -> Second (A) wins.
        # So Swapped[1] is prob that A wins.
        final_preds[:, 0] = 0.5 * (preds_orig[:, 0] + preds_swapped[:, 1])

        # Prob(B wins) = 0.5 * (Orig_B + Swapped_A_wins_which_is_index_0)
        # Swapped[0] is prob that B wins.
        final_preds[:, 1] = 0.5 * (preds_orig[:, 1] + preds_swapped[:, 0])

        # Prob(Tie)
        final_preds[:, 2] = 0.5 * (preds_orig[:, 2] + preds_swapped[:, 2])

        # --- Save Submission ---
        submission_df = pd.DataFrame(
            {
                "id": test_df["id"],
                "winner_model_a": final_preds[:, 0],
                "winner_model_b": final_preds[:, 1],
                "winner_tie": final_preds[:, 2],
            }
        )

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Validation metric {final_val_loss} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

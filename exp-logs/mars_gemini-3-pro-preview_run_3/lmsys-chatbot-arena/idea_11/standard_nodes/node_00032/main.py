import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import log_loss

# Import from provided library
from library.config import Config
from library.utils import seed_everything, get_logger, compute_metrics
from library.data_processing import get_dataloaders, SiameseDataset
from library.model import SiameseDeberta
from library.engine import train_fn, eval_fn, inference_fn

# Initialize Logger
logger = get_logger("runfile")


def analyze_failures(val_df, preds, targets):
    """
    Performs failure analysis by correlating error magnitude with features.
    """
    logger.info("Starting Failure Analysis...")

    # Calculate Log Loss per sample
    # Clip predictions to avoid log(0)
    epsilon = 1e-15
    preds = np.clip(preds, epsilon, 1 - epsilon)

    # targets are one-hot/probabilities. Calculate cross entropy per sample.
    # loss = - sum(target * log(pred))
    sample_losses = -np.sum(targets * np.log(preds), axis=1)

    # Extract features for correlation
    # We need lengths. Since we don't have the tokenized lengths readily available
    # without re-tokenizing, we'll use character/word lengths from the dataframe.
    val_df["loss"] = sample_losses
    val_df["len_prompt"] = val_df["prompt"].str.len()
    val_df["len_resp_a"] = val_df["response_a"].str.len()
    val_df["len_resp_b"] = val_df["response_b"].str.len()
    val_df["len_diff"] = (val_df["len_resp_a"] - val_df["len_resp_b"]).abs()

    # Calculate correlations
    correlations = {
        "len_prompt": val_df["loss"].corr(val_df["len_prompt"]),
        "len_resp_a": val_df["loss"].corr(val_df["len_resp_a"]),
        "len_resp_b": val_df["loss"].corr(val_df["len_resp_b"]),
        "len_diff": val_df["loss"].corr(val_df["len_diff"]),
    }

    logger.info("Correlation between Error (Log Loss) and Input Features:")
    for feature, corr in correlations.items():
        print(f"{feature}: {corr:.4f}")


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for faster execution within time limits
    # 3 epochs on augmented data (82k samples) might exceed 2 hours on some setups.
    # Reducing to 2 epochs to ensure safety while maintaining performance.
    Config.EPOCHS = 2

    logger.info(
        f"Effective Batch Size: {Config.TRAIN_BATCH_SIZE * Config.GRAD_ACCUM_STEPS}"
    )
    logger.info(f"Device: {Config.DEVICE}")

    # 2. Data Loading
    # We load cached data if available
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Model Initialization
    model = SiameseDeberta()
    model.to(Config.DEVICE)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    num_training_steps = len(train_loader) * Config.EPOCHS // Config.GRAD_ACCUM_STEPS
    num_warmup_steps = int(num_training_steps * Config.WARMUP_RATIO)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )

    # Scaler for FP16
    scaler = torch.amp.GradScaler("cuda", enabled=Config.FP16)

    # 5. Training Loop
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    logger.info("Starting Training...")
    for epoch in range(Config.EPOCHS):
        # Train
        train_loss = train_fn(
            model, train_loader, optimizer, scheduler, Config.DEVICE, scaler, epoch + 1
        )

        # Validate
        val_loss, val_preds = eval_fn(model, val_loader, Config.DEVICE)

        logger.info(
            f"Epoch {epoch+1} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}"
        )

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with loss {best_val_loss:.4f}")

    # 6. Final Evaluation & Failure Analysis
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.to(Config.DEVICE)

    final_val_loss, val_preds = eval_fn(model, val_loader, Config.DEVICE)

    # Print Metric exactly as requested
    print(f"Final Validation Metric: {final_val_loss}")

    # Load validation dataframe for analysis
    val_df = pd.read_parquet(os.path.join(Config.CACHE_DIR, "val_data.parquet"))
    # Ensure alignment (dataloaders might drop last if configured, but val usually doesn't)
    # The dataloader in library/data_processing.py does NOT drop_last for validation.
    # However, we should be careful about length.
    if len(val_df) != len(val_preds):
        logger.warning(
            f"Shape mismatch: DF {len(val_df)} vs Preds {len(val_preds)}. Truncating to min."
        )
        min_len = min(len(val_df), len(val_preds))
        val_df = val_df.iloc[:min_len]
        val_preds = val_preds[:min_len]

    targets = val_df[Config.TARGET_COLS].values
    analyze_failures(val_df, val_preds, targets)

    # 7. Submission
    THRESHOLD = 1.0005665522536111
    if final_val_loss < THRESHOLD:
        logger.info("Validation metric meets threshold. Generating submission...")

        # 7a. Prediction on Original Test Set (A, B)
        preds_orig = inference_fn(model, test_loader, Config.DEVICE)

        # 7b. Prediction on Swapped Test Set (B, A) - TTA
        logger.info("Generating swapped test set for TTA...")
        test_df = pd.read_parquet(os.path.join(Config.CACHE_DIR, "test_data.parquet"))

        swapped_test_df = test_df.copy()
        swapped_test_df["response_a"] = test_df["response_b"]
        swapped_test_df["response_b"] = test_df["response_a"]

        # Create temporary dataset/loader for swapped data
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_NAME)
        swapped_dataset = SiameseDataset(
            swapped_test_df, tokenizer, Config.MAX_LENGTH, is_test=True
        )
        swapped_loader = torch.utils.data.DataLoader(
            swapped_dataset,
            batch_size=Config.VALID_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        preds_swapped = inference_fn(model, swapped_loader, Config.DEVICE)

        # 7c. Average Predictions
        # preds_orig: [P(A>B), P(B>A), P(Tie)]
        # preds_swapped: [P(B>A), P(A>B), P(Tie)] (Input was B, A)

        # We want final: [P(A>B), P(B>A), P(Tie)]
        # P(A>B)_final = 0.5 * (P(A>B)_orig + P(A>B)_swapped_output[1])
        # P(B>A)_final = 0.5 * (P(B>A)_orig + P(B>A)_swapped_output[0])
        # P(Tie)_final = 0.5 * (P(Tie)_orig + P(Tie)_swapped_output[2])

        final_preds = np.zeros_like(preds_orig)
        final_preds[:, 0] = 0.5 * (preds_orig[:, 0] + preds_swapped[:, 1])  # Winner A
        final_preds[:, 1] = 0.5 * (preds_orig[:, 1] + preds_swapped[:, 0])  # Winner B
        final_preds[:, 2] = 0.5 * (preds_orig[:, 2] + preds_swapped[:, 2])  # Tie

        # 7d. Save Submission
        submission_df = pd.DataFrame(final_preds, columns=Config.TARGET_COLS)
        submission_df["id"] = test_df["id"]

        # Reorder columns to match sample submission: id, winner_model_a, winner_model_b, winner_tie
        cols = ["id"] + Config.TARGET_COLS
        submission_df = submission_df[cols]

        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.warning(
            f"Validation metric {final_val_loss} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

import sys
import os
import pandas as pd
import numpy as np
import torch
from library.config import Config
from library.utils import seed_everything, setup_logger, calculate_accuracy
from library.data_processor import get_data
from library.ngram_model import train_model, generate_submission


def main():
    # --- 1. Setup ---
    # Ensure reproducibility
    seed_everything(Config.SEED)
    logger = setup_logger("RunFile")

    # Detect hardware (Standard practice, though model is CPU-optimized)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Detected device: {device}")

    # --- 2. Train Model ---
    # train_model handles loading data and computing/loading cached N-gram stats
    logger.info("--- Training Model ---")
    model = train_model(load_cached_data=True)

    # --- 3. Validation & Evaluation ---
    logger.info("--- Validating Model ---")
    # Load validation data (cached if available)
    df_val = get_data("val", load_cached_data=True)

    # Generate predictions
    # The model expects the full sequence (with BOS/EOS) to use context,
    # and returns predictions for valid tokens only.
    preds = model.predict(df_val)

    # Align Ground Truth
    # Filter the validation dataframe to exclude BOS/EOS tokens to match predictions
    mask = (df_val[Config.INPUT_COL] != Config.BOS_TOKEN) & (
        df_val[Config.INPUT_COL] != Config.EOS_TOKEN
    )
    df_val_clean = df_val.loc[mask].copy()

    # Attach predictions
    df_val_clean["pred"] = preds

    # Prepare lists for metric calculation (ensure string type)
    y_true = df_val_clean[Config.TARGET_COL].astype(str).tolist()
    y_pred = df_val_clean["pred"].astype(str).tolist()

    # Calculate Accuracy
    acc = calculate_accuracy(y_true, y_pred)

    # REQUIRED: Print the final validation metric
    print(f"Final Validation Metric: {acc}")

    # --- 4. Failure Analysis ---
    logger.info("--- Failure Analysis ---")

    # Determine errors (0 = Correct, 1 = Error)
    df_val_clean["is_error"] = (
        df_val_clean[Config.TARGET_COL] != df_val_clean["pred"]
    ).astype(int)

    # Feature 1: Input Token Length
    df_val_clean["input_len"] = df_val_clean[Config.INPUT_COL].astype(str).str.len()

    # Feature 2: Sentence Length (Proxy for context complexity)
    # Count tokens per sentence in the clean dataframe
    sent_counts = df_val_clean.groupby(Config.SENTENCE_ID_COL).size().rename("sent_len")
    df_val_clean = df_val_clean.merge(
        sent_counts, on=Config.SENTENCE_ID_COL, how="left"
    )

    # Calculate Correlations
    corr_token_len = df_val_clean["input_len"].corr(df_val_clean["is_error"])
    corr_sent_len = df_val_clean["sent_len"].corr(df_val_clean["is_error"])

    print(f"Correlation (Input Length vs Error): {corr_token_len}")
    print(f"Correlation (Sentence Length vs Error): {corr_sent_len}")

    # Error Rate by Class
    if Config.CLASS_COL in df_val_clean.columns:
        print("\nError Rate by Class (Top 10 by count):")
        class_stats = df_val_clean.groupby(Config.CLASS_COL)["is_error"].agg(
            ["mean", "count"]
        )
        # Sort by count to see most impactful classes
        print(class_stats.sort_values("count", ascending=False).head(10))

        print("\nError Rate by Class (Top 10 by error rate, min 100 samples):")
        high_error_classes = class_stats[class_stats["count"] > 100].sort_values(
            "mean", ascending=False
        )
        print(high_error_classes.head(10))

    # --- 5. Submission ---
    logger.info("--- Generating Submission ---")
    generate_submission(model)
    logger.info("Process completed successfully.")


if __name__ == "__main__":
    main()

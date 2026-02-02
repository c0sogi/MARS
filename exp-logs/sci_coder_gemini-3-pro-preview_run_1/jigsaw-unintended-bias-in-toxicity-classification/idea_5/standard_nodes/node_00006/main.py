import sys
import os
import pandas as pd
import numpy as np
import torch
from scipy.stats import pearsonr

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, get_device
from library.dataset import get_dataloaders
from library.engine import run_training, predict_and_submit
from library.model import TriangulationDeberta
from library.metrics import calculate_score


def main():
    # ==========================================
    # 1. Setup & Configuration
    # ==========================================
    seed_everything(Config.SEED)
    logger = get_logger(name="main")

    # Override Config for Fast Baseline on A100
    # 1 Epoch is sufficient for a strong baseline with DeBERTa-base
    # Batch size 48 maximizes GPU utilization without OOM
    Config.EPOCHS = 1
    Config.TRAIN_BATCH_SIZE = 48
    Config.VALID_BATCH_SIZE = 128

    logger.info("Configuration configured for fast baseline execution.")
    logger.info(f"Epochs: {Config.EPOCHS}, Train Batch: {Config.TRAIN_BATCH_SIZE}")

    # ==========================================
    # 2. Data Loading
    # ==========================================
    logger.info("Loading dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.TRAIN_BATCH_SIZE,
        valid_batch_size=Config.VALID_BATCH_SIZE,
        load_cached_data=False,
    )

    # ==========================================
    # 3. Training
    # ==========================================
    logger.info("Starting training pipeline...")
    # run_training handles the training loop and saves the best model
    best_model_path = run_training(train_loader, val_loader)
    logger.info(f"Training complete. Best model saved at: {best_model_path}")

    # ==========================================
    # 4. Validation & Metric Calculation
    # ==========================================
    logger.info("Performing final validation inference...")
    device = get_device()

    # Load the best model for evaluation
    model = TriangulationDeberta(Config.MODEL_NAME)
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)
    model.eval()

    val_preds = []

    # Inference loop
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            outputs = model(input_ids, attention_mask)

            # Extract probabilities from the primary head
            probs = torch.sigmoid(outputs["primary"]).cpu().numpy().flatten()
            val_preds.extend(probs)

    # Load validation metadata to get targets and identity columns
    df_val = pd.read_csv(Config.VAL_META_PATH)

    # Ensure prediction length matches metadata (safety check)
    if len(val_preds) > len(df_val):
        val_preds = val_preds[: len(df_val)]
    elif len(val_preds) < len(df_val):
        # Should not happen with correct loaders, but pad if necessary
        val_preds.extend([0.0] * (len(df_val) - len(val_preds)))

    df_val["prediction"] = val_preds

    # Calculate Final Metric
    final_score, metrics_dict = calculate_score(df_val, "prediction")

    # REQUIRED: Print full precision metric
    print(f"Final Validation Metric: {final_score}")

    # ==========================================
    # 5. Failure Analysis
    # ==========================================
    logger.info("Performing failure analysis...")

    # Calculate absolute error
    df_val["error"] = (df_val["target"] - df_val["prediction"]).abs()

    # Calculate text length
    # Handle potential NaNs in text by treating them as empty strings
    df_val["text_len"] = df_val["comment_text"].fillna("").astype(str).apply(len)

    print("\n--- Failure Analysis (Correlation with Error Magnitude) ---")

    # 1. Correlation with Text Length
    if df_val["text_len"].std() > 0:
        corr_len, _ = pearsonr(df_val["text_len"], df_val["error"])
        print(f"Text Length: {corr_len:.4f}")
    else:
        print("Text Length: N/A (No variance)")

    # 2. Correlation with Identity Attributes
    # We check correlation between the presence of an identity (soft or hard) and error
    for ident in Config.IDENTITY_COLS:
        if ident in df_val.columns:
            # Fill NaNs with 0 (assuming NaN = identity not mentioned)
            ident_vals = df_val[ident].fillna(0.0)

            if ident_vals.std() > 0:
                corr, _ = pearsonr(ident_vals, df_val["error"])
                print(f"{ident}: {corr:.4f}")
            else:
                print(f"{ident}: N/A (No variance)")

    # ==========================================
    # 6. Submission Generation
    # ==========================================
    THRESHOLD = 0.9105405227619784

    if final_score > THRESHOLD:
        logger.info(
            f"Validation score ({final_score}) exceeds threshold ({THRESHOLD}). Generating submission..."
        )
        # predict_and_submit handles loading the model and saving the CSV
        predict_and_submit(best_model_path, test_loader)
    else:
        logger.info(
            f"Validation score ({final_score}) does not exceed threshold ({THRESHOLD}). Submission skipped."
        )


if __name__ == "__main__":
    main()

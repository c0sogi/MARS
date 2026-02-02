import os
import sys
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Ensure library modules can be imported
sys.path.append(os.getcwd())

from library.config import Config
from library.utils import seed_everything, get_logger
from library.data_processing import create_dataloaders
from library.model import ChatbotBiEncoder
from library.trainer import ModelTrainer

# Initialize Logger
logger = get_logger("runfile")


def perform_failure_analysis(model, val_loader):
    """
    Calculates per-sample loss on the validation set and correlates it with
    input features (text lengths) to identify error patterns.
    """
    logger.info("Starting Failure Analysis...")

    # Ensure model is in eval mode
    model.eval()
    device = Config.DEVICE

    # Store losses
    all_losses = []
    criterion = nn.CrossEntropyLoss(reduction="none")

    # 1. Compute Loss per sample
    with torch.no_grad():
        for features, targets in val_loader:
            if isinstance(features, dict):
                features = {k: v.to(device) for k, v in features.items()}
            else:
                features = features.to(device)

            targets = targets.to(device)

            logits = model(features)
            # Calculate loss per sample (no reduction)
            losses = criterion(logits, targets)
            all_losses.extend(losses.cpu().numpy())

    # 2. Load Validation Metadata to get features
    # We read the raw CSV to get text columns
    df_val = pd.read_csv(Config.VAL_DATA_PATH)

    # Safety check for length alignment
    if len(df_val) != len(all_losses):
        logger.warning(
            f"Size mismatch in failure analysis: DataFrame {len(df_val)} vs Predictions {len(all_losses)}"
        )
        min_len = min(len(df_val), len(all_losses))
        df_val = df_val.iloc[:min_len]
        all_losses = all_losses[:min_len]

    df_val["loss"] = all_losses

    # 3. Construct Features for Correlation
    # Calculate lengths of text inputs
    df_val["len_prompt"] = df_val["prompt"].fillna("").str.len()
    df_val["len_res_a"] = df_val["response_a"].fillna("").str.len()
    df_val["len_res_b"] = df_val["response_b"].fillna("").str.len()
    # Absolute difference in response lengths
    df_val["len_diff_abs"] = (df_val["len_res_a"] - df_val["len_res_b"]).abs()

    # 4. Calculate and Print Correlations
    features_to_check = ["len_prompt", "len_res_a", "len_res_b", "len_diff_abs"]

    print("\n" + "=" * 30)
    print("FAILURE ANALYSIS: Error Correlation")
    print("=" * 30)
    for feat in features_to_check:
        if feat in df_val.columns:
            corr = df_val[feat].corr(df_val["loss"])
            print(f"Correlation between {feat} and Loss: {corr:.10f}")
    print("=" * 30 + "\n")


def main():
    # --- 1. Configuration & Setup ---
    # Override Config for Fast Baseline
    Config.EPOCHS = 15  # Sufficient for MLP convergence on embeddings
    Config.BATCH_SIZE = 32  # Reduced from 512 to fit in 16GB GPU memory
    Config.EARLY_STOPPING_PATIENCE = 3

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    Config.setup()

    logger.info("Pipeline configuration complete.")

    # --- 2. Data Loading ---
    # Load cached embeddings or compute them if missing
    logger.info("Loading DataLoaders...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=Config.BATCH_SIZE, load_cached_data=True
    )

    # --- 3. Model Initialization ---
    logger.info(
        f"Initializing ChatbotBiEncoder (Fine-tuning {Config.TRANSFORMER_MODEL})"
    )

    model = ChatbotBiEncoder(
        model_name=Config.TRANSFORMER_MODEL,
        hidden_layers=Config.HIDDEN_LAYERS,
        dropout_rate=Config.DROPOUT_RATE,
        num_classes=Config.NUM_CLASSES,
    )

    # --- 4. Training ---
    trainer = ModelTrainer(model, train_loader, val_loader, test_loader)
    trainer.train()

    # --- 5. Validation Assessment ---
    # Load the best model saved during training
    if os.path.exists(Config.MODEL_SAVE_PATH):
        logger.info(f"Loading best model from {Config.MODEL_SAVE_PATH}")
        model.load_state_dict(
            torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
        )

    # Compute Final Validation Metric
    val_loss = trainer.validate()
    print(f"Final Validation Metric: {val_loss}")

    # --- 6. Failure Analysis ---
    perform_failure_analysis(model, val_loader)

    # --- 7. Submission ---
    trainer.generate_submission()
    logger.info("Submission generation complete.")


if __name__ == "__main__":
    main()

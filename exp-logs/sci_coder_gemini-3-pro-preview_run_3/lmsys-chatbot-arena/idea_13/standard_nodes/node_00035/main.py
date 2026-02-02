import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import log_loss
from scipy.stats import pearsonr

# Import from provided library files
from library.config import Config
from library.utils import seed_everything, get_logger
from library.dataset import get_dataloaders
from library.engine import run_training, generate_submission
from library.model import DualStreamSiameseModel

# Initialize logger
logger = get_logger("runfile", "runfile.log")


def validate_and_analyze(model_path, val_loader, device):
    """
    Loads the best model, performs inference on validation set,
    computes the official metric, and runs failure analysis.
    """
    logger.info("Starting Validation and Failure Analysis...")

    # 1. Load Model
    model = DualStreamSiameseModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []
    all_scalars = []

    # 2. Inference Loop
    # We replicate inference logic but also collect targets and scalars
    with torch.no_grad():
        for batch in val_loader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            type_a = batch["token_type_ids_a"].to(device)

            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            type_b = batch["token_type_ids_b"].to(device)

            scalars = batch["scalars"].to(device)
            targets = batch["target"].to(device)

            # Standard Forward Pass (No TTA for validation analysis to keep it clean,
            # though TTA is used for test. We stick to standard to analyze model bias).
            # Using mixed precision for speed
            with torch.cuda.amp.autocast(enabled=Config.USE_FP16):
                logits = model(
                    input_ids_a=ids_a,
                    attention_mask_a=mask_a,
                    token_type_ids_a=type_a,
                    input_ids_b=ids_b,
                    attention_mask_b=mask_b,
                    token_type_ids_b=type_b,
                    scalars=scalars,
                )

            probs = torch.softmax(logits, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_scalars.append(scalars.cpu().numpy())

    # Concatenate
    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_targets, axis=0)
    X_scalars = np.concatenate(all_scalars, axis=0)  # [log_len_p, log_len_a, log_len_b]

    # 3. Compute Metric
    # y_true are soft probabilities, log_loss supports this
    val_log_loss = log_loss(y_true, y_pred)

    print(f"Final Validation Metric: {val_log_loss}")

    # 4. Failure Analysis
    logger.info("Performing Failure Analysis...")

    # Calculate per-sample log loss (Cross Entropy)
    # loss = - sum(y_true * log(y_pred))
    # Clip preds for numerical stability manually or rely on small epsilon
    epsilon = 1e-15
    y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
    sample_losses = -np.sum(y_true * np.log(y_pred_clipped), axis=1)

    # Features
    # scalars: [log_len_prompt, log_len_resp_a, log_len_resp_b]
    log_len_p = X_scalars[:, 0]
    log_len_a = X_scalars[:, 1]
    log_len_b = X_scalars[:, 2]

    # Derived features
    len_diff = np.abs(log_len_a - log_len_b)

    # Correlations
    corr_p, _ = pearsonr(sample_losses, log_len_p)
    corr_diff, _ = pearsonr(sample_losses, len_diff)

    print("-" * 30)
    print("Failure Analysis Correlations (Error Magnitude vs Feature):")
    print(f"Prompt Length: {corr_p:.4f}")
    print(f"Response Length Difference: {corr_diff:.4f}")
    print("-" * 30)

    return val_log_loss


def main():
    # 1. Setup
    seed_everything(Config.SEED)

    # Override Config for Fast Baseline if necessary
    # A100 is fast, but we limit to 2 epochs to ensure we stay well within time limits
    Config.EPOCHS = 2

    logger.info(f"Initializing run with {Config.EPOCHS} epochs...")

    # 2. Data Loading
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=True)

    # 3. Training
    # This saves the best model to Config.BEST_MODEL_PATH
    if os.path.exists(Config.BEST_MODEL_PATH):
        logger.info(f"Model found at {Config.BEST_MODEL_PATH}. Skipping training.")
    else:
        run_training(train_loader, val_loader)

    # 4. Validation & Analysis
    val_metric = validate_and_analyze(Config.BEST_MODEL_PATH, val_loader, Config.DEVICE)

    # 5. Submission
    threshold = 1.0005665522536111
    if val_metric < threshold:
        logger.info(
            f"Validation metric {val_metric} < {threshold}. Generating submission..."
        )
        generate_submission(test_loader)
        logger.info("Submission generation complete.")
    else:
        logger.warning(
            f"Validation metric {val_metric} >= {threshold}. Skipping submission."
        )


if __name__ == "__main__":
    main()

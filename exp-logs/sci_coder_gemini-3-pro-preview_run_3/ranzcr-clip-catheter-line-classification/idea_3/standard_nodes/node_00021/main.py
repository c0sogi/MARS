import sys
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler

# Import from provided libraries
from library.config import Config
from library.utils import seed_everything, get_score, get_logger
from library.data import get_loaders, get_test_loader
from library.model import CatheterModel
from library.loss import CustomLoss
from library.train import train_one_epoch, valid_one_epoch


def main():
    # -------------------------------------------------------------------------
    # 1. Configuration Override for Fast Baseline
    # -------------------------------------------------------------------------
    # The A100 GPU allows for a larger batch size, removing the need for
    # gradient accumulation and speeding up training.
    Config.BATCH_SIZE = 16
    Config.EFFECTIVE_BATCH_SIZE = 16
    Config.GRADIENT_ACCUMULATION_STEPS = 1

    # Limit epochs to ensure completion within the time limit while allowing convergence
    # Increased to 10 to allow U-Net regularization to take effect (Cite solution_lesson_node_00008)
    Config.NUM_EPOCHS = 10

    # Ensure working directory exists
    Config.setup()

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Logger
    logger = get_logger("runfile")
    logger.info("Starting runfile execution...")

    device = torch.device(Config.DEVICE)
    logger.info(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 2. Data Loading
    # -------------------------------------------------------------------------
    logger.info("Loading DataLoaders...")
    train_loader, val_loader = get_loaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        load_cached_data=True,
        debug=Config.DEBUG,
    )

    # -------------------------------------------------------------------------
    # 3. Model & Training Setup
    # -------------------------------------------------------------------------
    logger.info("Initializing Model...")
    model = CatheterModel(pretrained=Config.PRETRAINED)
    model.to(device)

    criterion = CustomLoss(
        cls_weight=Config.CLS_LOSS_WEIGHT, aux_weight=Config.AUX_LOSS_WEIGHT
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.MIN_LR
    )

    scaler = GradScaler()

    # -------------------------------------------------------------------------
    # 4. Training Loop
    # -------------------------------------------------------------------------
    best_auc = 0.0

    logger.info(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        # Train
        train_loss = train_one_epoch(
            model, optimizer, criterion, train_loader, device, epoch, scaler
        )

        # Validation
        val_loss, val_auc = valid_one_epoch(model, criterion, val_loader, device)

        # Scheduler Step
        scheduler.step()

        elapsed = time.time() - start_time
        logger.info(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {elapsed:.0f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val AUC: {val_auc:.6f}"
        )

        # Save Best Model
        if val_auc > best_auc:
            best_auc = val_auc
            logger.info(f"New best AUC: {best_auc:.6f}. Saving model...")
            torch.save(model.state_dict(), Config.MODEL_PATH)

    # -------------------------------------------------------------------------
    # 5. Final Validation & Metric
    # -------------------------------------------------------------------------
    logger.info("Loading best model for final evaluation...")
    model.load_state_dict(torch.load(Config.MODEL_PATH))
    model.eval()

    val_preds = []
    val_targets = []

    # We perform inference manually here to get raw predictions for failure analysis
    with torch.no_grad():
        for images, labels, masks in val_loader:
            images = images.to(device)
            # Forward pass
            logits, _ = model(images)
            probs = torch.sigmoid(logits)

            val_preds.append(probs.cpu().numpy())
            val_targets.append(labels.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)

    # Calculate Final Metric
    final_metric = get_score(val_targets, val_preds)

    # PRINT REQUIRED METRIC
    print(f"Final Validation Metric: {final_metric}")

    # -------------------------------------------------------------------------
    # 6. Failure Analysis
    # -------------------------------------------------------------------------
    logger.info("Performing Failure Analysis...")

    # Calculate Mean Absolute Error (MAE) per sample across all classes
    # Shape: (N_samples,)
    mae_per_sample = np.mean(np.abs(val_targets - val_preds), axis=1)

    # Create a DataFrame for correlation analysis
    analysis_df = pd.DataFrame(val_targets, columns=Config.LABEL_COLS)
    analysis_df["error_magnitude"] = mae_per_sample

    print("\nCorrelation between Error Magnitude and Target Presence:")
    print("-" * 60)
    print(f"{'Label':<30} | {'Correlation':<12}")
    print("-" * 60)

    for label in Config.LABEL_COLS:
        # Calculate correlation between the binary label and the continuous error
        if analysis_df[label].nunique() > 1:
            corr = analysis_df[label].corr(analysis_df["error_magnitude"])
            print(f"{label:<30} | {corr:.6f}")
        else:
            print(f"{label:<30} | N/A (Constant)")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 7. Submission
    # -------------------------------------------------------------------------
    THRESHOLD = 0.9513452866855019

    if final_metric > THRESHOLD:
        logger.info("Threshold passed. Generating submission...")

        test_loader = get_test_loader(batch_size=Config.BATCH_SIZE)
        test_preds = []

        with torch.no_grad():
            for images, _, _ in test_loader:
                images = images.to(device)
                logits, _ = model(images)
                probs = torch.sigmoid(logits)
                test_preds.append(probs.cpu().numpy())

        test_preds = np.concatenate(test_preds, axis=0)

        # Load test metadata to get UIDs
        test_df = pd.read_csv(Config.TEST_METADATA_PATH)

        # Prepare Submission DataFrame
        sub_df = pd.DataFrame(test_preds, columns=Config.LABEL_COLS)
        sub_df.insert(0, "StudyInstanceUID", test_df["StudyInstanceUID"])

        # Save
        sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
        logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")

    else:
        logger.info(
            f"Threshold not reached ({final_metric} <= {THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()

import os
import sys
import torch
import numpy as np
import pandas as pd
import logging

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything, pf1_score, get_logger
from library.data_loader import get_dataloaders
from library.model import AsymmetryGatedSiameseNetwork
from library.training import run_training
from library.inference import predict_test_set

# Configure logging to print to stdout
logger = get_logger("demo_script")


def main():
    logger.info("Starting Demonstration Script...")

    # ------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides
    # ------------------------------------------------------------------------
    # We override Config values to ensure the demo runs quickly.
    # The Config class attributes are mutable.
    logger.info("Configuring environment for rapid demonstration...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Small sample size for speed
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Cite debug_lesson_30: Derive Data Dimensions from Configuration
    Config.IMAGE_SIZE = (256, 256)

    # Cite debug_lesson_32: Disable Multiprocessing DataLoaders in Debug Modes
    Config.NUM_WORKERS = 0

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    logger.info(f"Device: {Config.DEVICE}")
    logger.info(f"Debug Mode: {Config.DEBUG}")

    # ------------------------------------------------------------------------
    # 2. Data Loader Verification
    # ------------------------------------------------------------------------
    logger.info("\n[Step 1] Verifying Data Loaders...")

    train_loader, val_loader, test_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE,
        val_batch_size=Config.BATCH_SIZE,
        load_cached_data=False,  # Force reprocessing for demo
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
    )

    # Verify Train Loader
    try:
        batch = next(iter(train_loader))
        target_img, contra_img, labels = batch

        # Check Shapes
        # Expected: (B, 3, H, W) for images, (B,) for labels
        # Channels = 3 (Image + Age + Implant)
        expected_shape = (Config.BATCH_SIZE, 3, *Config.IMAGE_SIZE)

        assert (
            target_img.shape == expected_shape
        ), f"Target image shape mismatch. Expected {expected_shape}, got {target_img.shape}"
        assert (
            contra_img.shape == expected_shape
        ), f"Contralateral image shape mismatch. Expected {expected_shape}, got {contra_img.shape}"
        assert labels.shape == (
            Config.BATCH_SIZE,
        ), f"Labels shape mismatch. Expected {(Config.BATCH_SIZE,)}, got {labels.shape}"

        logger.info("Data Loader shapes verified successfully.")

    except StopIteration:
        logger.error("Train loader is empty!")
        raise

    # ------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # ------------------------------------------------------------------------
    logger.info("\n[Step 2] Verifying Model Architecture...")

    model = AsymmetryGatedSiameseNetwork()
    model = model.to(Config.DEVICE)

    # Move batch to device
    target_img = target_img.to(Config.DEVICE)
    contra_img = contra_img.to(Config.DEVICE)

    # Perform Forward Pass
    with torch.no_grad():
        logits = model(target_img, contra_img)

    # Verify Output Shape
    # Expected: (B, 1)
    assert logits.shape == (
        Config.BATCH_SIZE,
        1,
    ), f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 1)}, got {logits.shape}"

    logger.info("Model forward pass successful. Output shape verified.")

    # ------------------------------------------------------------------------
    # 4. Training Loop Execution
    # ------------------------------------------------------------------------
    logger.info("\n[Step 3] Running Training Loop (1 Epoch)...")

    save_path = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    trained_model = run_training(
        num_epochs=Config.NUM_EPOCHS,
        batch_size=Config.BATCH_SIZE,
        learning_rate=1e-4,
        pos_weight=1.0,  # Simplified weight for demo
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
        patience=1,
        save_path=save_path,
    )

    # Verify Model Checkpoint
    if os.path.exists(save_path):
        logger.info(f"Training complete. Model saved to {save_path}")
    else:
        # Note: If validation loss doesn't improve (which can happen with random init and 1 epoch),
        # the model might not be saved by the logic in training.py.
        # However, training.py saves if val_loss < best_val_loss (inf).
        # So it should save at least once after epoch 1.
        raise FileNotFoundError(
            f"Model checkpoint not found at {save_path} after training."
        )

    # ------------------------------------------------------------------------
    # 5. Inference Execution
    # ------------------------------------------------------------------------
    logger.info("\n[Step 4] Running Inference on Test Set...")

    submission_path = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    df_submission = predict_test_set(
        model_path=save_path,
        batch_size=Config.BATCH_SIZE,
        device=Config.DEVICE,
        debug=Config.DEBUG,
        debug_sample_size=Config.DEBUG_SAMPLE_SIZE,
        save_path=submission_path,
    )

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created."
    assert not df_submission.empty, "Submission DataFrame is empty."
    assert "prediction_id" in df_submission.columns, "Missing 'prediction_id' column."
    assert "cancer" in df_submission.columns, "Missing 'cancer' column."

    # Verify values are probabilities
    probs = df_submission["cancer"].values
    assert np.all(
        (probs >= 0) & (probs <= 1)
    ), "Predictions contain values outside [0, 1]."

    logger.info(f"Inference successful. Generated {len(df_submission)} predictions.")

    # ------------------------------------------------------------------------
    # 6. Metric Verification
    # ------------------------------------------------------------------------
    logger.info("\n[Step 5] Verifying Metric Calculation (pF1)...")

    # Create dummy data
    # Case: Perfect prediction
    labels_perfect = np.array([0, 1, 0, 1])
    preds_perfect = np.array([0.0, 1.0, 0.0, 1.0])
    score_perfect = pf1_score(labels_perfect, preds_perfect)

    # Case: Random prediction
    labels_mixed = np.array([0, 1, 0, 1])
    preds_mixed = np.array([0.2, 0.8, 0.4, 0.6])
    score_mixed = pf1_score(labels_mixed, preds_mixed)

    logger.info(f"pF1 Score (Perfect): {score_perfect:.4f}")
    logger.info(f"pF1 Score (Mixed): {score_mixed:.4f}")

    assert score_perfect > 0.99, "pF1 calculation for perfect predictions failed."
    assert 0 <= score_mixed <= 1, "pF1 score out of range."

    logger.info("\n==== Demonstration Completed Successfully ====")


if __name__ == "__main__":
    main()

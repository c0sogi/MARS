import os
import sys
import torch
import numpy as np
import pandas as pd
import glob

# Import from the provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    get_logger,
    calibrate_probabilities,
    pf1_score,
)
from library.dataset import get_dataloaders
from library.model import MilTransformerModel
from library.trainer import Trainer


def main():
    # --------------------------------------------------------------------------
    # 1. Setup and Configuration Overrides for Speed
    # --------------------------------------------------------------------------
    print(">>> 1. Configuring environment for fast demonstration...")

    # Set seeds for reproducibility
    seed_everything(Config.SEED)

    # Override Config for a quick demo run
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 20  # Use a tiny subset
    Config.BATCH_SIZE = 2
    Config.NUM_EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small data
    Config.PRETRAINED = False  # Disable download for speed/offline safety

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    logger = get_logger("demo_script")
    logger.info("Configuration updated for demo.")

    # --------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # --------------------------------------------------------------------------
    print("\n>>> 2. Initializing DataLoaders...")

    # This function handles metadata reading, caching, and loader creation
    train_loader, val_loader, test_loader = get_dataloaders(load_cached_data=False)

    # Verify DataLoaders are not empty
    assert len(train_loader) > 0, "Train loader is empty!"
    assert len(val_loader) > 0, "Val loader is empty!"
    assert len(test_loader) > 0, "Test loader is empty!"

    # Fetch one batch to verify structure
    print("Verifying batch structure...")
    images_list, targets = next(iter(train_loader))

    # Check Inputs
    # images_list is a list of tensors (one tensor per bag/breast)
    assert isinstance(images_list, list), "Expected images_list to be a list"
    assert (
        len(images_list) == Config.BATCH_SIZE
    ), f"Expected batch size {Config.BATCH_SIZE}"
    # Check first image tensor: (Num_Views, 3, H, W)
    # Num_Views is variable, but usually >= 1
    assert images_list[0].ndim == 4, "Image tensor should be 4D (Views, C, H, W)"
    assert images_list[0].shape[1] == 3, "Expected 3 channels (RGB)"

    # Check Targets
    assert isinstance(targets, dict), "Expected targets to be a dict"
    assert "cancer" in targets
    assert "density" in targets
    assert "biopsy" in targets
    assert targets["cancer"].shape[0] == Config.BATCH_SIZE

    logger.info("DataLoaders and batch structure verified.")

    # --------------------------------------------------------------------------
    # 3. Model Initialization and Forward Pass
    # --------------------------------------------------------------------------
    print("\n>>> 3. Initializing Model and checking forward pass...")

    model = MilTransformerModel()
    model.to(Config.DEVICE)

    # Perform forward pass with the batch fetched earlier
    # Move data to device
    images_list_device = [img.to(Config.DEVICE) for img in images_list]

    with torch.no_grad():
        outputs = model(images_list_device)

    # Verify Outputs
    assert "cancer" in outputs
    assert "density" in outputs
    assert "biopsy" in outputs

    # Check shapes
    # Cancer: (B, 1)
    assert outputs["cancer"].shape == (Config.BATCH_SIZE, 1)
    # Density: (B, 4)
    assert outputs["density"].shape == (Config.BATCH_SIZE, 4)
    # Biopsy: (B, 1)
    assert outputs["biopsy"].shape == (Config.BATCH_SIZE, 1)

    logger.info("Model forward pass successful. Output shapes correct.")

    # --------------------------------------------------------------------------
    # 4. Training Loop Demonstration
    # --------------------------------------------------------------------------
    print("\n>>> 4. Running Training Loop (1 Epoch)...")

    trainer = Trainer(model, train_loader, val_loader)

    # Run fit (train + validate + save)
    trainer.fit()

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_SAVE_PATH
    ), "Model checkpoint not found after training!"
    logger.info(f"Training complete. Model saved to {Config.MODEL_SAVE_PATH}")

    # --------------------------------------------------------------------------
    # 5. Inference and Calibration Demonstration
    # --------------------------------------------------------------------------
    print("\n>>> 5. Running Inference on Test Set...")

    # Load the best model
    model.load_state_dict(
        torch.load(Config.MODEL_SAVE_PATH, map_location=Config.DEVICE)
    )
    model.eval()

    predictions = []
    prediction_ids = []

    with torch.no_grad():
        for images_list, bag_ids in test_loader:
            images_list = [img.to(Config.DEVICE) for img in images_list]

            # Forward
            out = model(images_list)

            # Get raw logits for cancer
            logits = out["cancer"]

            # Sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            predictions.extend(probs)
            prediction_ids.extend(bag_ids)

    predictions = np.array(predictions)

    # Verify we have predictions
    assert len(predictions) > 0
    assert len(predictions) == len(prediction_ids)

    # Apply Calibration
    # Shift probability distribution to match expected test prevalence
    calibrated_preds = calibrate_probabilities(
        predictions,
        train_prevalence=Config.TRAIN_PREVALENCE,
        test_prevalence=Config.TEST_PREVALENCE,
    )

    # Verify calibration effect (simple check: mean should likely decrease as test prev is low)
    logger.info(f"Raw Mean Prob: {predictions.mean():.4f}")
    logger.info(f"Calibrated Mean Prob: {calibrated_preds.mean():.4f}")

    # Create Submission DataFrame
    submission_df = pd.DataFrame(
        {"prediction_id": prediction_ids, "cancer": calibrated_preds}
    )

    # Save submission
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {Config.SUBMISSION_PATH}")
    assert os.path.exists(Config.SUBMISSION_PATH)

    # --------------------------------------------------------------------------
    # 6. Metric Verification (pF1 Score)
    # --------------------------------------------------------------------------
    print("\n>>> 6. Verifying pF1 Score Metric...")

    # Create dummy data
    # Case 1: Perfect prediction
    y_true = np.array([0, 0, 1, 1])
    y_pred_perfect = np.array([0.0, 0.0, 1.0, 1.0])
    score_perfect = pf1_score(y_true, y_pred_perfect)

    # Case 2: Random prediction
    y_pred_random = np.array([0.5, 0.5, 0.5, 0.5])
    score_random = pf1_score(y_true, y_pred_random)

    logger.info(f"pF1 Score (Perfect): {score_perfect:.4f}")
    logger.info(f"pF1 Score (Random): {score_random:.4f}")

    assert score_perfect > 0.99, "Perfect predictions should yield pF1 ~ 1.0"
    assert score_perfect > score_random, "Perfect prediction should beat random"

    print("\n>>> Demo Completed Successfully.")


if __name__ == "__main__":
    main()

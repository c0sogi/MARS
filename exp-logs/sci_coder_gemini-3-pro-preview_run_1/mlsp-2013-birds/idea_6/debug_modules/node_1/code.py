import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import shutil

# Import from the provided library files
from library.config import Config
from library.utils import set_seed, Logger
from library.dataset import get_data_loaders
from library.model import BirdResNet
from library.engine import train_one_epoch, validate, inference


def main():
    # --- 1. Setup and Configuration ---
    print("--- Setting up Demo Configuration ---")

    # Set a fixed seed for reproducibility
    set_seed(42)

    # Modify Config for a fast demo run
    # We redirect the working directory to avoid overwriting main experiment files
    Config.WORKING_DIR = "./working/demo_run"
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Update cache paths to the new working directory
    Config.CACHE_TRAIN_IMAGES = os.path.join(Config.WORKING_DIR, "train_images.npy")
    Config.CACHE_TRAIN_LABELS = os.path.join(Config.WORKING_DIR, "train_labels.npy")
    Config.CACHE_VAL_IMAGES = os.path.join(Config.WORKING_DIR, "val_images.npy")
    Config.CACHE_VAL_LABELS = os.path.join(Config.WORKING_DIR, "val_labels.npy")
    Config.CACHE_TEST_IMAGES = os.path.join(Config.WORKING_DIR, "test_images.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.WORKING_DIR, "test_ids.npy")

    Config.TEACHER_MODEL_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Enable Debug mode to use a tiny subset of data (e.g., 32 samples)
    Config.DEBUG = True
    Config.DEBUG_SUBSET_SIZE = 32
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo
    Config.TEACHER_EPOCHS = 1  # Only run 1 epoch

    # Initialize Logger
    logger = Logger(os.path.join(Config.WORKING_DIR, "demo.log"))
    logger.log("Configuration updated for demo run.")

    # --- 2. Data Loading Demonstration ---
    logger.log("\n--- Loading Data ---")

    # get_data_loaders handles processing, caching, and loader creation
    # We set load_cached_data=False to force processing logic demonstration
    train_loader, val_loader, test_loader = get_data_loaders(
        Config, pseudo_labels=None, load_cached_data=False
    )

    # Verification: Check Loader Sizes
    logger.log(f"Train Batches: {len(train_loader)}")
    logger.log(f"Val Batches: {len(val_loader)}")
    logger.log(f"Test Batches: {len(test_loader)}")

    assert len(train_loader) > 0, "Train loader is empty."

    # Verification: Check Input Shape (Batch, Channels, Height, Width)
    # Channels should be 3 (Intensity + Delta + Delta-Delta)
    sample_imgs, sample_lbls = next(iter(train_loader))
    logger.log(f"Sample Input Shape: {sample_imgs.shape}")
    logger.log(f"Sample Label Shape: {sample_lbls.shape}")

    assert (
        sample_imgs.shape[1] == 3
    ), f"Expected 3 input channels, got {sample_imgs.shape[1]}"
    assert sample_imgs.shape[2] == Config.IMG_HEIGHT, "Incorrect Image Height"
    assert sample_imgs.shape[3] == Config.IMG_WIDTH, "Incorrect Image Width"
    assert sample_lbls.shape[1] == Config.NUM_CLASSES, "Incorrect Label Dimension"

    # --- 3. Model Instantiation ---
    logger.log("\n--- Initializing Model ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"Device: {device}")

    model = BirdResNet(num_classes=Config.NUM_CLASSES, pretrained=True)
    model.to(device)

    # Verification: Dummy Forward Pass
    dummy_input = torch.randn(2, 3, Config.IMG_HEIGHT, Config.IMG_WIDTH).to(device)
    with torch.no_grad():
        dummy_output = model(dummy_input)

    logger.log(f"Model Output Shape: {dummy_output.shape}")
    assert dummy_output.shape == (2, Config.NUM_CLASSES), "Model output shape mismatch"

    # --- 4. Training Loop Demonstration ---
    logger.log("\n--- Starting Training Demo (1 Epoch) ---")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, device, Config)
    logger.log(f"Training Loss: {train_loss:.4f}")

    assert not np.isnan(train_loss), "Training loss is NaN"

    # --- 5. Validation Demonstration ---
    logger.log("\n--- Starting Validation Demo ---")

    val_loss, val_auc = validate(model, val_loader, device)
    logger.log(f"Validation Loss: {val_loss:.4f}")
    logger.log(f"Validation AUC: {val_auc:.4f}")

    # Save the model (demonstrating utility function)
    torch.save(model.state_dict(), Config.TEACHER_MODEL_PATH)
    logger.log("Model checkpoint saved.")

    # --- 6. Inference Demonstration ---
    logger.log("\n--- Starting Inference Demo ---")

    ids, probs = inference(model, test_loader, device)

    logger.log(f"Inference IDs Shape: {ids.shape}")
    logger.log(f"Inference Probs Shape: {probs.shape}")

    assert len(ids) == len(probs), "Mismatch between IDs and Predictions"
    assert probs.shape[1] == Config.NUM_CLASSES, "Incorrect prediction classes"

    # --- 7. Submission Formatting ---
    logger.log("\n--- Formatting Submission ---")

    # The submission format requires flattening the predictions:
    # Id = rec_id * 100 + species_id
    # Probability = prob

    submission_rows = []
    for i in range(len(ids)):
        rec_id = int(ids[i])
        rec_probs = probs[i]

        for species_id in range(Config.NUM_CLASSES):
            row_id = rec_id * 100 + species_id
            prob = rec_probs[species_id]
            submission_rows.append({"Id": row_id, "Probability": prob})

    submission_df = pd.DataFrame(submission_rows)

    # Save submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission.csv")
    submission_df.to_csv(submission_path, index=False)

    logger.log(f"Submission saved to {submission_path}")
    logger.log(f"Submission Head:\n{submission_df.head()}")

    print("\n--- Demo Completed Successfully ---")


if __name__ == "__main__":
    main()

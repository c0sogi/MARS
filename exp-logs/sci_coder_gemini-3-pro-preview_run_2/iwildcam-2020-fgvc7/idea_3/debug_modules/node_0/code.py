import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, load_megadetector_data
from library.data_loader import get_dataloaders, IWildCamDataset
from library.model import IWildCamModel
from library.trainer import Trainer


def main():
    print("=== Starting Library Demonstration & Verification ===")

    # --------------------------------------------------------------------------
    # 1. Configuration Setup & Override
    # --------------------------------------------------------------------------
    print("\n[1] Setting up Configuration...")
    # Initialize directories and seeds
    Config.setup()

    # Override Config parameters for a fast demonstration (Debug Mode)
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.DEBUG_SAMPLE_SIZE = 64  # Use only 64 images for train/val/test
    Config.BATCH_SIZE = 8  # Reduce batch size
    Config.NUM_WORKERS = 2  # Adjust workers for the environment

    print(
        f"    Debug Config: Epochs={Config.EPOCHS}, "
        f"Batch Size={Config.BATCH_SIZE}, "
        f"Sample Size={Config.DEBUG_SAMPLE_SIZE}"
    )

    # --------------------------------------------------------------------------
    # 2. Utility Verification
    # --------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test loading MegaDetector bounding boxes
    # This function caches results to parquet, speeding up future loads
    bbox_dict = load_megadetector_data()

    # Assertions
    if not isinstance(bbox_dict, dict):
        raise TypeError("load_megadetector_data should return a dictionary.")
    if len(bbox_dict) == 0:
        raise ValueError("Bounding box dictionary is empty.")

    print(f"    Successfully loaded {len(bbox_dict)} bounding box entries.")

    # --------------------------------------------------------------------------
    # 3. Data Loader Verification
    # --------------------------------------------------------------------------
    print("\n[3] Verifying Data Loaders...")

    # Initialize DataLoaders in debug mode (uses the subsampled size defined above)
    train_loader, val_loader, test_loader, mixup_fn = get_dataloaders(debug=True)

    # Verify Train Loader Batch
    try:
        images, targets = next(iter(train_loader))
    except StopIteration:
        raise RuntimeError("Train loader is empty.")

    print(
        f"    Train Batch - Image Shape: {images.shape}, Target Shape: {targets.shape}"
    )

    # Assertions for shapes
    expected_shape = (Config.BATCH_SIZE, 3, Config.IMG_SIZE, Config.IMG_SIZE)
    if images.shape != expected_shape:
        raise AssertionError(
            f"Expected image shape {expected_shape}, got {images.shape}"
        )

    if targets.shape != (Config.BATCH_SIZE,):
        raise AssertionError(
            f"Expected target shape {(Config.BATCH_SIZE,)}, got {targets.shape}"
        )

    # Verify Mixup Functionality
    if mixup_fn is not None:
        mixed_images, mixed_targets = mixup_fn(images, targets)
        # Mixup targets should be (Batch, Num_Classes) because of one-hot/smoothing
        if mixed_targets.shape != (Config.BATCH_SIZE, Config.NUM_CLASSES):
            raise AssertionError(
                f"Mixup target shape mismatch. Expected {(Config.BATCH_SIZE, Config.NUM_CLASSES)}, got {mixed_targets.shape}"
            )
        print("    Mixup transformation verified.")

    # --------------------------------------------------------------------------
    # 4. Model Architecture Verification
    # --------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    # Instantiate model (using pretrained=False for speed in this demo)
    model = IWildCamModel(pretrained=False)
    model.eval()

    # Run a dummy forward pass
    dummy_input = torch.randn(2, 3, Config.IMG_SIZE, Config.IMG_SIZE).to(Config.DEVICE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"    Forward pass output shape: {output.shape}")

    # Assertions
    if output.shape != (2, Config.NUM_CLASSES):
        raise AssertionError(
            f"Model output shape mismatch. Expected (2, {Config.NUM_CLASSES}), got {output.shape}"
        )

    # Clean up memory
    del model, dummy_input, output
    torch.cuda.empty_cache()

    # --------------------------------------------------------------------------
    # 5. Training Loop Verification
    # --------------------------------------------------------------------------
    print("\n[5] Verifying Training Loop (Trainer)...")

    # Initialize Trainer with debug=True
    # This will use the overridden Config values (1 Epoch, small dataset)
    trainer = Trainer(debug=True)

    # Execute training
    print("    Starting fit()...")
    trainer.fit()

    # Verify artifact generation
    if not os.path.exists(Config.BEST_MODEL_PATH):
        raise FileNotFoundError(f"Best model was not saved to {Config.BEST_MODEL_PATH}")

    print(f"    Training complete. Model saved to {Config.BEST_MODEL_PATH}")

    # --------------------------------------------------------------------------
    # 6. Inference & Submission Logic Verification
    # --------------------------------------------------------------------------
    print("\n[6] Verifying Inference Logic...")

    # Load the trained model
    inference_model = IWildCamModel(pretrained=False)
    inference_model.load_state_dict(
        torch.load(Config.BEST_MODEL_PATH, map_location=Config.DEVICE)
    )
    inference_model.to(Config.DEVICE)
    inference_model.eval()

    predictions = []

    # Iterate through the test loader (subset)
    print("    Running inference on test subset...")
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(Config.DEVICE)
            outputs = inference_model(images)

            # Get predicted class indices
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            predictions.extend(preds)

    num_preds = len(predictions)
    print(f"    Generated {num_preds} predictions.")

    # Verify we have predictions for the debug subset
    # Note: DataLoader might drop last batch if drop_last=True, but test loader usually doesn't.
    # We just check if we got any results.
    if num_preds == 0:
        raise RuntimeError("No predictions generated during inference verification.")

    print("\n=== All Verification Steps Passed Successfully ===")


if __name__ == "__main__":
    main()

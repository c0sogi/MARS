import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings
import shutil

# Filter warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library components
from library.config import Config
from library.utils import seed_everything, get_device
from library.dataset import ArtworkDataset, get_dataloaders, get_test_loader
from library.model import ArtworkConvNeXt
from library.loss import AsymmetricLoss
from library.train import run_training
from library.optimize import main_optimization_pipeline


def demo_pipeline():
    print("=== Starting Artwork Attribute Labeling Demo ===")

    # --- 1. Patch Configuration for Speed ---
    print("\n[1] Configuring environment for rapid demonstration...")

    # Modify Config attributes directly to control the execution flow
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 64  # Small subset for speed
    Config.NUM_EPOCHS = 1  # Only 1 epoch
    Config.BATCH_SIZE = 8  # Small batch size
    Config.NUM_WORKERS = 2  # Reduce workers to avoid overhead
    Config.WORKING_DIR = "./working/demo_run"  # Separate demo directory
    Config.MODEL_PATH = os.path.join(Config.WORKING_DIR, "demo_model.pth")

    # Re-run setup to create the new working directory
    Config.setup()

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Sample Size: {Config.DEBUG_SAMPLE_SIZE}")
    print(f"    Working Dir: {Config.WORKING_DIR}")

    # Set seeds
    seed_everything(Config.SEED)
    device = get_device()
    print(f"    Device: {device}")

    # --- 2. Verify Dataset and DataLoader ---
    print("\n[2] Verifying Dataset and DataLoader...")

    train_loader, val_loader = get_dataloaders(
        debug=Config.DEBUG, batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Fetch one batch
    images, targets = next(iter(train_loader))

    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Target Batch Shape: {targets.shape}")

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect image tensor shape"
    assert targets.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect target tensor shape"
    assert targets.dtype == torch.float32, "Targets should be float32"
    print("    Dataset verification passed.")

    # --- 3. Verify Model and Loss ---
    print("\n[3] Verifying Model Architecture and Loss Function...")

    # Instantiate Model
    model = ArtworkConvNeXt(
        model_name=Config.MODEL_NAME,
        pretrained=False,  # False for speed in demo (avoids downloading weights if not cached)
        num_classes=Config.NUM_CLASSES,
    )
    model.to(device)

    # Forward pass with the batch from step 2
    images = images.to(device)
    targets = targets.to(device)

    logits = model(images)
    print(f"    Logits Shape: {logits.shape}")

    assert logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Model output shape mismatch"

    # Instantiate Loss
    criterion = AsymmetricLoss()
    loss = criterion(logits, targets)

    print(f"    Calculated Loss: {loss.item():.4f}")

    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() >= 0, "Loss should be non-negative"
    print("    Model and Loss verification passed.")

    # --- 4. Run Training Loop ---
    print("\n[4] Running Training Loop (1 Epoch, Debug Subset)...")

    # This calls the Trainer class internally
    run_training(debug=Config.DEBUG)

    # Verify model checkpoint was saved
    assert os.path.exists(
        Config.MODEL_PATH
    ), f"Model checkpoint not found at {Config.MODEL_PATH}"
    print(f"    Training complete. Model saved to {Config.MODEL_PATH}")

    # --- 5. Run Optimization Pipeline ---
    print("\n[5] Running Threshold Optimization...")

    # This generates validation predictions and finds optimal thresholds
    # We force load_cached_data=False to ensure the code runs fully
    best_thresholds = main_optimization_pipeline(load_cached_data=False)

    assert (
        len(best_thresholds) == Config.NUM_CLASSES
    ), "Thresholds array length mismatch"
    assert os.path.exists(
        os.path.join(Config.WORKING_DIR, "best_thresholds.npy")
    ), "Thresholds file not saved"
    print("    Optimization complete.")

    # --- 6. Simulate Inference and Submission ---
    print("\n[6] Simulating Inference on Test Set...")

    # Load Test Loader
    test_loader = get_test_loader(
        batch_size=Config.BATCH_SIZE, num_workers=Config.NUM_WORKERS
    )

    # Load Model Weights
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))
    model.eval()

    # Inference on a single batch for demonstration
    test_images, test_ids = next(iter(test_loader))
    test_images = test_images.to(device)

    with torch.no_grad():
        test_logits = model(test_images)
        test_probs = torch.sigmoid(test_logits).cpu().numpy()

    # Apply Optimized Thresholds
    # Broadcasting: (Batch, Classes) >= (Classes,)
    predictions_binary = (test_probs >= best_thresholds[None, :]).astype(int)

    # Format Output
    submission_rows = []
    for i in range(len(test_ids)):
        img_id = test_ids[i]
        # Get indices where prediction is 1
        pred_indices = np.where(predictions_binary[i] == 1)[0]
        pred_str = " ".join(map(str, pred_indices))
        submission_rows.append({"id": img_id, "attribute_ids": pred_str})

    df_submission = pd.DataFrame(submission_rows)
    print("    Generated Submission Sample:")
    print(df_submission.head())

    assert len(df_submission) == Config.BATCH_SIZE, "Submission sample size mismatch"
    assert (
        "id" in df_submission.columns and "attribute_ids" in df_submission.columns
    ), "Submission columns missing"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    demo_pipeline()

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import provided library components
from library.config import Config, seed_everything
from library.data import get_dataloaders, get_test_dataloader
from library.model import SequenceSmoothedMIL
from library.loss import HierarchicalCompoundLoss
from library.utils import competition_log_loss
from library.train import train_model


def main():
    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore")

    print("=== RSNA Cervical Spine Fracture Detection: Pipeline Demonstration ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Override
    # ------------------------------------------------------------------------
    # We modify the global Config class to optimize for a quick demonstration run.
    print("[1] Configuring environment for rapid demonstration...")

    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 12  # Small subset for quick iteration
    Config.EPOCHS = 1  # Single epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.SEQ_LEN = 16  # Reduced sequence length to save memory/time
    Config.NUM_WORKERS = 0  # Disable multiprocessing to avoid overhead in demo
    Config.WORKING_DIR = "./working/demo_run"  # Separate directory for demo artifacts

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seeds for reproducibility
    seed_everything(Config.SEED)
    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Sequence Length: {Config.SEQ_LEN}")

    # ------------------------------------------------------------------------
    # 2. Data Loading Demonstration
    # ------------------------------------------------------------------------
    print("\n[2] Demonstrating Data Loading...")

    # Initialize Dataloaders
    train_loader, val_loader = get_dataloaders(
        train_metadata_path=Config.TRAIN_METADATA,
        val_metadata_path=Config.VAL_METADATA,
        image_dir=Config.TRAIN_IMAGES_DIR,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # Fetch a single batch to verify shapes
    print("    Fetching a batch from train_loader...")
    images, labels = next(iter(train_loader))

    print(f"    Images Shape: {images.shape} (Batch, Seq, Channels, H, W)")
    print(f"    Labels Shape: {labels.shape} (Batch, Classes)")

    # Assertions to verify data integrity
    expected_img_shape = (
        Config.BATCH_SIZE,
        Config.SEQ_LEN,
        3,
        Config.IMG_SIZE,
        Config.IMG_SIZE,
    )
    expected_lbl_shape = (Config.BATCH_SIZE, 8)

    if images.shape != expected_img_shape:
        raise AssertionError(
            f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
        )
    if labels.shape != expected_lbl_shape:
        raise AssertionError(
            f"Label shape mismatch. Expected {expected_lbl_shape}, got {labels.shape}"
        )

    print("    Data loading verification passed.")

    # ------------------------------------------------------------------------
    # 3. Model Architecture Demonstration
    # ------------------------------------------------------------------------
    print("\n[3] Demonstrating Model Architecture...")

    # Instantiate the model
    # We set pretrained=False here to ensure the demo runs even if internet is restricted,
    # though the main training script defaults to True.
    model = SequenceSmoothedMIL(
        backbone_name="resnet18", pretrained=False, num_classes=7
    )
    model.to(Config.DEVICE)
    model.eval()

    # Perform a forward pass
    print("    Performing forward pass...")
    with torch.no_grad():
        logits = model(images.to(Config.DEVICE))

    print(f"    Logits Shape: {logits.shape}")

    # Verify output shape
    if logits.shape != (Config.BATCH_SIZE, 8):
        raise AssertionError(
            f"Model output shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {logits.shape}"
        )

    print("    Model forward pass verification passed.")

    # ------------------------------------------------------------------------
    # 4. Loss Function Demonstration
    # ------------------------------------------------------------------------
    print("\n[4] Demonstrating Loss Function...")

    criterion = HierarchicalCompoundLoss()

    # Calculate loss on the batch
    loss = criterion(logits, labels.to(Config.DEVICE))
    print(f"    Calculated Loss: {loss.item():.6f}")

    # Verify loss is a valid scalar
    if torch.isnan(loss) or loss.item() < 0:
        raise AssertionError("Loss is NaN or negative.")

    print("    Loss calculation verification passed.")

    # ------------------------------------------------------------------------
    # 5. Metric Demonstration
    # ------------------------------------------------------------------------
    print("\n[5] Demonstrating Competition Metric (Weighted Log Loss)...")

    # Create synthetic ground truth and predictions
    # Case: Patient has C1 fracture (implies patient_overall=1)
    y_true_demo = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 1],  # Fracture at C1
            [0, 0, 0, 0, 0, 0, 0, 0],  # Healthy
        ]
    )

    # Predictions: Confident correct prediction for row 0, unsure for row 1
    y_pred_demo = np.array(
        [
            [0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.9],
            [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2],
        ]
    )

    metric_val = competition_log_loss(y_true_demo, y_pred_demo)
    print(f"    Metric Value: {metric_val:.6f}")

    if not isinstance(metric_val, float):
        raise AssertionError("Metric did not return a float.")

    print("    Metric calculation verification passed.")

    # ------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[6] Demonstrating Full Training Loop...")
    print("    Running 'train_model' with overridden config (1 Epoch, Debug Subset)...")

    # train_model uses the global Config we modified earlier
    best_model_path = train_model(epochs=Config.EPOCHS)

    print(f"    Training finished. Best model path: {best_model_path}")

    if not os.path.exists(best_model_path):
        raise AssertionError(f"Model file was not created at {best_model_path}")

    print("    Training loop verification passed.")

    # ------------------------------------------------------------------------
    # 7. Inference Demonstration
    # ------------------------------------------------------------------------
    print("\n[7] Demonstrating Inference Setup...")

    test_loader, test_df = get_test_dataloader(
        Config.TEST_METADATA,
        Config.TEST_IMAGES_DIR,
        Config.BATCH_SIZE,
        Config.NUM_WORKERS,
    )

    print(f"    Test DataFrame loaded with {len(test_df)} studies.")

    # Attempt to fetch one batch from test loader
    try:
        test_images, _ = next(iter(test_loader))
        print(f"    Test Batch Shape: {test_images.shape}")
    except StopIteration:
        print("    Test loader is empty (check dataset availability).")

    print("    Inference setup verification passed.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()

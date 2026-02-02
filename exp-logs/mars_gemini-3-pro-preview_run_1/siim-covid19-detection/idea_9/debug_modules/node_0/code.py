import sys
import os
import torch
import pandas as pd
import numpy as np
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import from the provided library files
from library.config import Config
from library.utils import seed_everything
from library.data import get_train_val_loaders, get_test_loader
from library.model import MultiTaskModel
from library.loss import MultiTaskLoss
from library.engine import train_one_epoch, evaluate, inference_and_submit


def main():
    print(
        "Starting demonstration of the SIIM-FISABIO-RSNA COVID-19 Detection pipeline..."
    )

    # 1. Setup and Configuration Overrides for Demo
    print("\n[1] Configuring environment for rapid demonstration...")
    seed_everything(Config.SEED)

    # Override Config for speed
    Config.DEBUG = True  # Use a small subset of data
    Config.EPOCHS = 1  # Run only 1 epoch
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in demo

    # Ensure working directory exists (handled by Config, but good to double check context)
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    print(f"    Debug Mode: {Config.DEBUG}")
    print(f"    Batch Size: {Config.BATCH_SIZE}")
    print(f"    Device: {Config.DEVICE}")

    # 2. Data Loading Verification
    print("\n[2] initializing Data Loaders...")
    # We set load_cached_data=False to demonstrate the preprocessing logic
    train_loader, val_loader = get_train_val_loaders(load_cached_data=False)

    print(f"    Train Loader length: {len(train_loader)}")
    print(f"    Val Loader length: {len(val_loader)}")

    # Verify Batch Structure
    images, labels, masks = next(iter(train_loader))
    print(
        f"    Batch Shapes -> Images: {images.shape}, Labels: {labels.shape}, Masks: {masks.shape}"
    )

    # Assertions
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Image Shape"
    assert labels.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Label Shape"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Mask Shape"
    print("    Data Loader verification passed.")

    # 3. Model Initialization and Forward Pass
    print("\n[3] Initializing Model...")
    # Use pretrained=False to speed up initialization (no download needed if not cached)
    model = MultiTaskModel(pretrained=False)
    model.to(Config.DEVICE)

    print("    Performing dummy forward pass...")
    images = images.to(Config.DEVICE)
    cls_logits, seg_logits = model(images)

    print(
        f"    Output Shapes -> Cls Logits: {cls_logits.shape}, Seg Logits: {seg_logits.shape}"
    )

    # Assertions
    assert cls_logits.shape == (
        Config.BATCH_SIZE,
        Config.NUM_CLASSES,
    ), "Incorrect Classification Output Shape"
    assert seg_logits.shape == (
        Config.BATCH_SIZE,
        1,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    ), "Incorrect Segmentation Output Shape"
    print("    Model verification passed.")

    # 4. Loss Function Verification
    print("\n[4] Testing Loss Function...")
    criterion = MultiTaskLoss()

    # Move targets to device
    labels = labels.to(Config.DEVICE)
    masks = masks.to(Config.DEVICE)

    loss_dict = criterion(cls_logits, seg_logits, labels, masks)

    print(
        f"    Loss Values -> Total: {loss_dict['loss'].item():.4f}, "
        f"Cls: {loss_dict['cls_loss'].item():.4f}, Seg: {loss_dict['seg_loss'].item():.4f}"
    )

    assert not torch.isnan(loss_dict["loss"]), "Loss is NaN"
    print("    Loss function verification passed.")

    # 5. Training Loop Demonstration
    print("\n[5] Running Training Loop (1 Epoch)...")
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    # No scheduler for this short demo
    scheduler = None

    avg_train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, Config.DEVICE, epoch=0
    )
    print(f"    Average Train Loss: {avg_train_loss:.4f}")
    assert avg_train_loss > 0, "Train loss should be positive"

    # 6. Evaluation Demonstration
    print("\n[6] Running Evaluation...")
    val_loss, val_map = evaluate(model, val_loader, Config.DEVICE)
    print(f"    Validation Loss: {val_loss:.4f}")
    print(f"    Validation mAP: {val_map:.4f}")

    assert val_loss > 0, "Validation loss should be positive"
    assert 0.0 <= val_map <= 1.0, "mAP score out of range [0, 1]"

    # 7. Inference and Submission
    print("\n[7] Running Inference and Generating Submission...")
    test_loader = get_test_loader(load_cached_data=False)

    # Define output path for submission
    submission_path = os.path.join(Config.WORKING_DIR, "submission_demo.csv")

    inference_and_submit(model, test_loader, Config.DEVICE, output_path=submission_path)

    # Verify Submission File
    assert os.path.exists(submission_path), "Submission file was not created"

    sub_df = pd.read_csv(submission_path)
    print(f"    Submission file created at {submission_path}")
    print(f"    Rows: {len(sub_df)}")
    print(f"    Columns: {list(sub_df.columns)}")

    # Check format
    assert (
        "id" in sub_df.columns and "PredictionString" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    # Check content format (simple check on first row)
    first_pred = sub_df.iloc[0]["PredictionString"]
    print(f"    Sample Prediction: {first_pred}")
    assert isinstance(first_pred, str), "PredictionString is not a string"

    print("\nDemonstration completed successfully!")


if __name__ == "__main__":
    main()

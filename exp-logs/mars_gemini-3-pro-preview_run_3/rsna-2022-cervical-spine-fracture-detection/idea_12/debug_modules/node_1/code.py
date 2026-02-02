import os
import sys
import torch
import pandas as pd
import numpy as np
import logging

# Import library modules
from library.config import Config
from library.utils import seed_everything, get_logger, load_dicom
from library.data import get_dataloaders, RSNADataset
from library.model import RSNAModel
from library.loss import ImplicitWeightedLoss
from library.train import run_training
from library.inference import predict
import library.model  # Imported to patch the class method


def main():
    print(
        "=== Starting Demonstration of RSNA Cervical Spine Fracture Detection Pipeline ==="
    )

    # -------------------------------------------------------------------------
    # 1. Configuration Overrides for Demo Speed
    # -------------------------------------------------------------------------
    print("\n[1] Configuring environment for fast demonstration...")

    # Modify Config to run a very lightweight version of the task
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.DEBUG_SAMPLE_SIZE = 4  # Use only 4 samples for training/val/test
    Config.NUM_SLICES = 8  # Reduce sequence length from 64 to 8 for speed
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple debug run

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Monkey-patch RSNAModel.__init__ to force pretrained=False.
    # This prevents connection errors if the environment restricts downloads.
    original_init = library.model.RSNAModel.__init__

    def patched_init(self, pretrained=True):
        # We ignore the requested 'pretrained' value and force False for this demo
        original_init(self, pretrained=False)

    library.model.RSNAModel.__init__ = patched_init
    print(
        "    Config configured: Debug=True, Batch=2, Slices=8, Pretrained=False (Patched)"
    )

    # -------------------------------------------------------------------------
    # 2. Utilities Demonstration
    # -------------------------------------------------------------------------
    print("\n[2] Testing Utilities...")
    seed_everything(42)
    logger = get_logger(
        "demo_logger", log_file=os.path.join(Config.WORKING_DIR, "demo.log")
    )
    logger.info("Logger initialized successfully.")

    # Test DICOM loading
    # We look up a valid image path from the metadata
    train_meta = pd.read_csv(Config.TRAIN_METADATA_PATH)
    if not train_meta.empty:
        sample_row = train_meta.iloc[0]
        sample_dir = os.path.join(Config.INPUT_DIR, sample_row["image_path"])

        # Find the first .dcm file in the directory
        dcm_files = [f for f in os.listdir(sample_dir) if f.endswith(".dcm")]
        if dcm_files:
            dcm_path = os.path.join(sample_dir, dcm_files[0])
            dcm = load_dicom(dcm_path)
            if dcm is not None:
                print(f"    Successfully loaded DICOM: {dcm_path}")
                print(f"    Pixel Array Shape: {dcm.pixel_array.shape}")
            else:
                raise ValueError("Failed to load DICOM file.")
        else:
            print("    No DICOM files found in the sample directory to test.")
    else:
        print("    Train metadata is empty.")

    # -------------------------------------------------------------------------
    # 3. Data Pipeline Demonstration
    # -------------------------------------------------------------------------
    print("\n[3] Testing Data Pipeline (Dataset & DataLoader)...")

    # Get dataloaders with debug flag (loads small subset)
    train_loader, val_loader = get_dataloaders(
        train_batch_size=Config.BATCH_SIZE, debug=True
    )

    print(f"    Train Loader Batches: {len(train_loader)}")

    # Fetch one batch to verify shapes
    images, targets = next(iter(train_loader))

    # Expected Shapes:
    # Images: (Batch, Seq, Channels, H, W) -> (2, 8, 3, 224, 224)
    expected_img_shape = (
        Config.BATCH_SIZE,
        Config.NUM_SLICES,
        3,
        Config.IMAGE_SIZE,
        Config.IMAGE_SIZE,
    )
    # Targets: (Batch, 8 classes)
    expected_target_shape = (Config.BATCH_SIZE, 8)

    print(f"    Image Batch Shape: {images.shape}")
    print(f"    Target Batch Shape: {targets.shape}")

    assert (
        images.shape == expected_img_shape
    ), f"Image shape mismatch. Expected {expected_img_shape}, got {images.shape}"
    assert (
        targets.shape == expected_target_shape
    ), f"Target shape mismatch. Expected {expected_target_shape}, got {targets.shape}"
    print("    Data shapes verified.")

    # -------------------------------------------------------------------------
    # 4. Model Architecture Demonstration
    # -------------------------------------------------------------------------
    print("\n[4] Testing Model Architecture...")
    device = torch.device("cpu")  # Use CPU for this quick shape check

    # Initialize model (uses patched init, so pretrained=False)
    model = RSNAModel(pretrained=False)
    model.to(device)
    model.eval()

    with torch.no_grad():
        # Forward pass
        logits = model(images.to(device))

    print(f"    Logits Shape: {logits.shape}")
    assert logits.shape == (
        Config.BATCH_SIZE,
        8,
    ), f"Output shape mismatch. Expected {(Config.BATCH_SIZE, 8)}, got {logits.shape}"
    print("    Forward pass successful.")

    # -------------------------------------------------------------------------
    # 5. Loss Function Demonstration
    # -------------------------------------------------------------------------
    print("\n[5] Testing Loss Function...")
    criterion = ImplicitWeightedLoss()
    loss = criterion(logits, targets.to(device))

    print(f"    Calculated Loss: {loss.item():.4f}")
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"
    print("    Loss calculation verified.")

    # -------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop (1 Epoch, Debug Mode)...")

    # Run training using the library function
    # We explicitly pass the modified config values because default args are bound at definition time
    try:
        run_training(
            epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, debug=True, patience=1
        )
        print("    Training loop completed successfully.")
    except Exception as e:
        print(f"    Training loop failed: {e}")
        raise e

    # Check if model checkpoint was saved
    # Note: If validation loss doesn't improve (possible with random init/small data),
    # best_model.pth might not be saved. We handle this for the inference step.
    if os.path.exists(Config.MODEL_SAVE_PATH):
        print(f"    Checkpoint saved at: {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "    Note: best_model.pth not created (Validation loss might not have improved)."
        )
        # Create a dummy checkpoint for the inference step to succeed
        torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
        print("    Created dummy checkpoint for inference testing.")

    # -------------------------------------------------------------------------
    # 7. Inference Demonstration
    # -------------------------------------------------------------------------
    print("\n[7] Running Inference (Debug Mode)...")

    try:
        predict(debug=True, batch_size=Config.BATCH_SIZE)
        print("    Inference completed successfully.")
    except Exception as e:
        print(f"    Inference failed: {e}")
        raise e

    # Verify submission file
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"    Submission file generated at: {Config.SUBMISSION_PATH}")
        print(f"    Submission Rows: {len(sub_df)}")
        print("    Sample Predictions:")
        print(sub_df.head().to_string())

        # Validation
        assert "row_id" in sub_df.columns, "row_id column missing in submission"
        assert "fractured" in sub_df.columns, "fractured column missing in submission"
    else:
        raise FileNotFoundError("Submission file was not generated.")

    print("\n=== Demonstration Completed Successfully ===")


if __name__ == "__main__":
    main()

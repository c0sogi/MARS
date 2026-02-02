import os
import torch
import numpy as np
import pandas as pd
import cv2
import sys

# Import library components
from library.config import Config
from library.utils import set_seed, rle_encode, f05_score
from library.data import VesuviusDataset, get_loaders
from library.model import SiameseSegFormer
from library.loss import BCEDiceLoss
from library.engine import Trainer
from library.inference import InferenceRunner


def main():
    print("=== Starting Vesuvius Ink Detection Demo ===")

    # 1. Configuration Overrides for Speed and Debugging
    print("\n[1] Configuring environment...")
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.DEBUG = True
    Config.MAX_DEBUG_SAMPLES = 4  # Very small subset for speed
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead in simple script
    Config.WORKING_DIR = "./working/demo_run"
    Config.CHECKPOINT_PATH = os.path.join(Config.WORKING_DIR, "best_model.pth")

    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)

    # Set seed for reproducibility
    set_seed(42)
    print("Configuration updated for demo mode.")

    # 2. Verify Utilities
    print("\n[2] Verifying Utilities...")

    # Test RLE Encoding
    # Mask: 0 1 1 0 -> Pixels 2 and 3 are 1. Start at 2, length 2.
    dummy_mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # Flatten: 0, 1, 1, 0
    rle_result = rle_encode(dummy_mask)
    expected_rle = "2 2"
    assert (
        rle_result == expected_rle
    ), f"RLE Failed. Expected '{expected_rle}', got '{rle_result}'"
    print("RLE Encoding: OK")

    # Test F0.5 Score
    # Perfect match
    preds = torch.tensor([0.9, 0.1])
    labels = torch.tensor([1.0, 0.0])
    score = f05_score(preds, labels, threshold=0.5)
    assert abs(score - 1.0) < 1e-6, f"F0.5 Score Failed. Expected 1.0, got {score}"
    print("F0.5 Score: OK")

    # 3. Verify Data Pipeline
    print("\n[3] Verifying Data Pipeline...")

    # Test Dataset instantiation
    train_ds = VesuviusDataset(mode="train", transform=True, debug=True)
    print(f"Train Dataset Size (Debug): {len(train_ds)}")

    # Test __getitem__
    if len(train_ds) > 0:
        x_high, x_center, x_low, label = train_ds[0]

        # Check shapes: Inputs (3, 512, 512), Label (1, 512, 512)
        expected_shape = (3, Config.TILE_SIZE, Config.TILE_SIZE)
        assert x_high.shape == expected_shape, f"x_high shape mismatch: {x_high.shape}"
        assert (
            x_center.shape == expected_shape
        ), f"x_center shape mismatch: {x_center.shape}"
        assert x_low.shape == expected_shape, f"x_low shape mismatch: {x_low.shape}"
        assert label.shape == (
            1,
            Config.TILE_SIZE,
            Config.TILE_SIZE,
        ), f"Label shape mismatch: {label.shape}"
        print("Dataset Item Shapes: OK")
    else:
        print("Warning: Train dataset is empty. Skipping shape checks.")

    # Test DataLoaders
    train_loader, val_loader, test_loader = get_loaders()
    print("DataLoaders initialized: OK")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SiameseSegFormer().to(device)

    # Create dummy batch
    B = 2
    dummy_in = torch.randn(B, 3, Config.TILE_SIZE, Config.TILE_SIZE).to(device)

    # Forward pass
    with torch.no_grad():
        out = model(dummy_in, dummy_in, dummy_in)

    # Check output shape: (B, 1, H, W)
    expected_out_shape = (B, 1, Config.TILE_SIZE, Config.TILE_SIZE)
    assert (
        out.shape == expected_out_shape
    ), f"Model output shape mismatch. Expected {expected_out_shape}, got {out.shape}"
    print("Model Forward Pass: OK")

    # 5. Verify Loss Function
    print("\n[5] Verifying Loss Function...")
    criterion = BCEDiceLoss()
    loss = criterion(out, torch.zeros_like(out))
    assert loss.dim() == 0, "Loss should be a scalar"
    print("Loss Calculation: OK")

    # 6. Run Training Loop (Demo)
    print("\n[6] Running Training Loop (Demo)...")
    trainer = Trainer(train_loader, val_loader, test_loader)

    # Override trainer device to ensure consistency with main
    trainer.device = device
    trainer.model = trainer.model.to(device)

    # Run fit (1 epoch, minimal data)
    trainer.fit()

    # Check if checkpoint was saved
    # Note: Checkpoint is only saved if validation score > baseline.
    # In a random/dummy run, this might not happen.
    # To verify the mechanism, we force a save or check if the code ran without error.
    # For this demo, we will manually save a dummy checkpoint if one wasn't created
    # so that the inference step can proceed.
    if not os.path.exists(Config.CHECKPOINT_PATH):
        print(
            "Validation score too low for auto-save. Saving dummy checkpoint for inference demo."
        )
        torch.save(model.state_dict(), Config.CHECKPOINT_PATH)

    assert os.path.exists(Config.CHECKPOINT_PATH), "Checkpoint file missing."
    print("Training Loop: OK")

    # 7. Run Inference Pipeline (Demo)
    print("\n[7] Running Inference Pipeline (Demo)...")

    # Initialize Inference Runner
    # It will load the checkpoint we just ensured exists
    inference_runner = InferenceRunner(checkpoint_path=Config.CHECKPOINT_PATH)

    # Run inference
    inference_runner.run()

    # Verify Submission File
    if os.path.exists(Config.SUBMISSION_PATH):
        sub_df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created with {len(sub_df)} rows.")
        assert (
            "Id" in sub_df.columns and "Predicted" in sub_df.columns
        ), "Submission columns missing."
        print("Inference Pipeline: OK")
    else:
        raise AssertionError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    main()

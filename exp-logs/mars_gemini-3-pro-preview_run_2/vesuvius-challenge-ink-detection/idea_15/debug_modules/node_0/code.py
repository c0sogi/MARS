import os
import sys
import numpy as np
import pandas as pd
import torch
import cv2

# Import from the provided library files
from library.config import Config
from library.utils import rle_encoding, fbeta_score, BCEDiceLoss, seed_everything
from library.dataset import InkDataset
from library.model import HybridSegFormerUNet
from library.trainer import TrainingEngine
from library.inference import InferenceEngine


def run_demo():
    print("=== Setting up Configuration for Demo ===")
    # Modify Config to run a fast, lightweight demo
    Config.EPOCHS = 1
    Config.SAMPLE_SIZE = 10  # Use only 10 samples for training/validation
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = (
        0  # Use 0 workers to avoid multiprocessing complexity in script
    )
    Config.PRETRAINED = False  # Disable downloading pretrained weights
    Config.SCAN_OFFSETS = [0]  # Use single Z-offset for inference speed
    Config.IDEA_NAME = "demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, Config.IDEA_NAME)
    Config.CHECKPOINT_DIR = Config.CACHE_DIR
    Config.SUBMISSION_PATH = os.path.join(Config.WORKING_DIR, "demo_submission.csv")

    # Ensure working directories exist
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    # Set seed for reproducibility
    seed_everything(Config.SEED)
    print("Configuration updated for speed and reproducibility.")

    # -------------------------------------------------------------------------
    # 1. Demonstrate Utilities
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating Utilities ===")

    # Test RLE Encoding
    # Create a simple 1D binary mask: 0 1 1 1 0 0 1 0
    # 1-based indexing:
    # 1s at indices 2,3,4 (start 2, len 3) and index 7 (start 7, len 1)
    dummy_mask = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    rle_result = rle_encoding(dummy_mask)
    expected_rle = "2 3 7 1"
    print(f"RLE Input: {dummy_mask}")
    print(f"RLE Output: {rle_result}")
    assert (
        rle_result == expected_rle
    ), f"RLE mismatch: got {rle_result}, expected {expected_rle}"

    # Test Metrics (F-Beta)
    preds = torch.tensor([0.1, 0.9, 0.8, 0.2])
    targets = torch.tensor([0.0, 1.0, 0.0, 1.0])
    # Threshold 0.5 -> Preds: [0, 1, 1, 0], Targets: [0, 1, 0, 1]
    # TP=1 (idx 1), FP=1 (idx 2), FN=1 (idx 3)
    # Beta=0.5 -> (1.25 * TP) / (1.25*TP + 0.25*FN + FP)
    # (1.25 * 1) / (1.25 + 0.25 + 1) = 1.25 / 2.5 = 0.5
    score = fbeta_score(preds, targets, beta=0.5, threshold=0.5)
    print(f"F0.5 Score: {score}")
    assert np.isclose(score, 0.5), f"F-Beta score mismatch: got {score}"

    # Test Loss
    loss_fn = BCEDiceLoss()
    loss = loss_fn(
        torch.randn(2, 1, 32, 32), torch.randint(0, 2, (2, 1, 32, 32)).float()
    )
    print(f"BCEDiceLoss: {loss.item()}")
    assert not torch.isnan(loss), "Loss is NaN"

    # -------------------------------------------------------------------------
    # 2. Demonstrate Dataset
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating InkDataset ===")

    # Initialize Train Dataset
    # This will trigger slab processing and caching (might take a moment)
    train_ds = InkDataset(
        Config.TRAIN_METADATA_PATH, mode="train", limit_size=Config.SAMPLE_SIZE
    )
    print(f"Train Dataset Length: {len(train_ds)}")
    assert len(train_ds) == Config.SAMPLE_SIZE, "Dataset did not respect limit_size"

    # Fetch one sample
    img, label = train_ds[0]
    print(f"Train Sample Image Shape: {img.shape}")
    print(f"Train Sample Label Shape: {label.shape}")

    assert img.shape == (3, Config.TILE_SIZE, Config.TILE_SIZE), "Incorrect image shape"
    assert label.shape == (
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect label shape"
    assert isinstance(img, torch.Tensor), "Image is not a tensor"

    # Initialize Test Dataset
    test_ds = InkDataset(
        Config.TEST_METADATA_PATH, mode="test", limit_size=Config.SAMPLE_SIZE
    )
    t_img, t_coord, t_fid = test_ds[0]
    print(f"Test Sample Image Shape: {t_img.shape}")
    print(f"Test Sample Coord: {t_coord}, Fragment ID: {t_fid}")
    assert t_img.shape == (
        3,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Incorrect test image shape"

    # -------------------------------------------------------------------------
    # 3. Demonstrate Model
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating HybridSegFormerUNet ===")
    model = HybridSegFormerUNet()
    # Mock input batch
    dummy_input = torch.randn(2, 3, Config.TILE_SIZE, Config.TILE_SIZE)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Model Input Shape: {dummy_input.shape}")
    print(f"Model Output Shape: {output.shape}")

    assert output.shape == (
        2,
        1,
        Config.TILE_SIZE,
        Config.TILE_SIZE,
    ), "Model output shape mismatch"

    # -------------------------------------------------------------------------
    # 4. Demonstrate Training Engine
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating TrainingEngine ===")
    trainer = TrainingEngine()

    # Run one epoch
    print("Running single training epoch...")
    train_loss = trainer.train_one_epoch(0)
    print(f"Epoch 0 Train Loss: {train_loss}")
    assert isinstance(train_loss, float), "Train loss is not a float"

    # Run validation
    print("Running validation...")
    val_score = trainer.validate()
    print(f"Validation F0.5 Score: {val_score}")
    assert 0.0 <= val_score <= 1.0, "Validation score out of range"

    # Save checkpoint
    # Force save by setting best_score to -1 temporarily or just calling save
    prev_best = trainer.best_score
    trainer.best_score = -1.0
    trainer.save_checkpoint(val_score)
    expected_ckpt = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    assert os.path.exists(expected_ckpt), "Checkpoint file was not created"
    print(f"Checkpoint saved at: {expected_ckpt}")

    # -------------------------------------------------------------------------
    # 5. Demonstrate Inference Engine
    # -------------------------------------------------------------------------
    print("\n=== Demonstrating InferenceEngine ===")
    # Initialize inference engine with the checkpoint we just saved
    inference_engine = InferenceEngine(checkpoint_path=expected_ckpt)

    # Generate submission
    print("Generating submission (this involves processing test fragments)...")
    inference_engine.generate_submission()

    assert os.path.exists(Config.SUBMISSION_PATH), "Submission file not created"

    # Verify Submission Content
    sub_df = pd.read_csv(Config.SUBMISSION_PATH)
    print("Submission File Head:")
    print(sub_df.head())

    assert (
        "Id" in sub_df.columns and "Predicted" in sub_df.columns
    ), "Submission columns missing"
    assert len(sub_df) > 0, "Submission file is empty"

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

import os
import sys
import torch
import numpy as np
import pandas as pd
import warnings

# Import from the provided library
from library.config import (
    GeneralConfig,
    PathConfig,
    TrainConfig,
    DataConfig,
    seed_everything,
)
from library.dataset import get_loaders, get_test_loader
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.trainer import ModelTrainer
from library.inference import Evaluator
from library.utils import rle_encode, rle_decode

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Salt Segmentation Demo ===\n")

    # 1. Configuration Overrides for Demo Speed
    print("1. Configuring environment for rapid execution...")

    # Set reproducible seed
    seed_everything(GeneralConfig.SEED)

    # Modify paths to isolate this run
    PathConfig.WORKING_DIR = "./working/demo_execution"
    PathConfig.CHECKPOINT_DIR = os.path.join(PathConfig.WORKING_DIR, "checkpoints")
    PathConfig.LOG_DIR = os.path.join(PathConfig.WORKING_DIR, "logs")
    PathConfig.SUBMISSION_DIR = os.path.join(PathConfig.WORKING_DIR, "submission")
    PathConfig.CACHE_DIR = os.path.join(PathConfig.WORKING_DIR, "cache")
    PathConfig.create_directories()

    # Reduce training complexity
    GeneralConfig.DEBUG = True
    GeneralConfig.DEBUG_DATA_LIMIT = 50  # Use only 50 samples
    TrainConfig.EPOCHS = 2  # 1 Warmup, 1 Finetune
    TrainConfig.WARMUP_EPOCHS = 1
    TrainConfig.EARLY_STOPPING_PATIENCE = 2
    DataConfig.BATCH_SIZE = 4  # Small batch size

    print(f"   Working Directory: {PathConfig.WORKING_DIR}")
    print(f"   Debug Mode: {GeneralConfig.DEBUG}")
    print(f"   Epochs: {TrainConfig.EPOCHS}")

    # 2. Data Loading Verification
    print("\n2. Verifying Data Loading...")
    # Force load_cached_data=False to ensure we test the raw loading logic
    train_loader, val_loader = get_loaders(
        fold_idx=0, debug=True, load_cached_data=False
    )

    # Fetch one batch
    images, masks, ids = next(iter(train_loader))

    print(f"   Batch Image Shape: {images.shape}")  # Should be (B, 3, 128, 128)
    print(f"   Batch Mask Shape: {masks.shape}")  # Should be (B, 1, 128, 128)

    # Assertions
    assert images.shape == (
        DataConfig.BATCH_SIZE,
        3,
        DataConfig.IMG_H,
        DataConfig.IMG_W,
    ), "Incorrect Image Shape"
    assert masks.shape == (
        DataConfig.BATCH_SIZE,
        1,
        DataConfig.IMG_H,
        DataConfig.IMG_W,
    ), "Incorrect Mask Shape"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"

    # Check normalization (approximate check)
    assert (
        images.max() <= 10.0 and images.min() >= -10.0
    ), "Image values out of expected normalized range"
    print("   Data Loading Verified.")

    # 3. Model Logic Verification
    print("\n3. Verifying Model Architecture...")
    device = torch.device(GeneralConfig.DEVICE)
    model = SaltUNetPlusPlus().to(device)

    # Move batch to device
    images = images.to(device)

    # Test Training Forward Pass (Deep Supervision)
    model.train()
    outputs_train = model(images)
    print(f"   Training Output Type: {type(outputs_train)}")
    print(f"   Number of Deep Supervision Heads: {len(outputs_train)}")
    assert isinstance(outputs_train, list), "Model should return list in training mode"
    assert len(outputs_train) == 4, "Should have 4 output heads"
    assert outputs_train[-1].shape == (
        DataConfig.BATCH_SIZE,
        1,
        DataConfig.IMG_H,
        DataConfig.IMG_W,
    )

    # Test Eval Forward Pass
    model.eval()
    with torch.no_grad():
        output_eval = model(images)
    print(f"   Eval Output Shape: {output_eval.shape}")
    assert isinstance(
        output_eval, torch.Tensor
    ), "Model should return Tensor in eval mode"
    assert output_eval.shape == (
        DataConfig.BATCH_SIZE,
        1,
        DataConfig.IMG_H,
        DataConfig.IMG_W,
    )
    print("   Model Architecture Verified.")

    # 4. Loss Function Verification
    print("\n4. Verifying Loss Functions...")
    masks = masks.to(device)

    # BCEDiceLoss (Warmup)
    criterion_warmup = BCEDiceLoss()
    # Use one of the deep supervision outputs
    loss_val_warmup = criterion_warmup(outputs_train[0], masks)
    print(f"   BCE+Dice Loss: {loss_val_warmup.item():.4f}")
    assert not torch.isnan(loss_val_warmup), "BCE+Dice Loss is NaN"
    assert loss_val_warmup > 0, "BCE+Dice Loss should be positive"

    # LovaszHingeLoss (Finetune)
    criterion_finetune = LovaszHingeLoss()
    loss_val_lovasz = criterion_finetune(outputs_train[-1], masks)
    print(f"   Lovasz Loss: {loss_val_lovasz.item():.4f}")
    assert not torch.isnan(loss_val_lovasz), "Lovasz Loss is NaN"
    print("   Loss Functions Verified.")

    # 5. Training Loop Execution
    print("\n5. Running Training Loop (Fold 0)...")
    # We use the Trainer class which encapsulates the loop
    trainer = ModelTrainer(fold_idx=0, debug=True)

    # Run training
    checkpoint_path = trainer.run()

    print(f"   Training finished. Checkpoint saved at: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), "Checkpoint file was not created"

    # 6. Inference & Optimization
    print("\n6. Running Inference and Threshold Optimization...")
    evaluator = Evaluator(device=device)

    # Generate OOF predictions
    # Note: We use load_cached_data=False to force generation
    preds, targets = evaluator.predict_fold(
        fold_idx=0, model_path=checkpoint_path, load_cached_data=False, debug=True
    )

    print(f"   OOF Predictions Shape: {preds.shape}")
    print(f"   OOF Targets Shape: {targets.shape}")

    # Verify cropping to original size (101x101)
    assert preds.shape[-2:] == (
        DataConfig.ORIG_H,
        DataConfig.ORIG_W,
    ), "Predictions not cropped to original size"

    # Optimize Threshold
    best_threshold = evaluator.optimize_threshold(preds, targets)
    print(f"   Best Threshold Found: {best_threshold}")
    assert 0.0 < best_threshold < 1.0, "Threshold out of bounds"

    # 7. Submission Generation
    print("\n7. Generating Submission...")
    sub_df = evaluator.generate_submission(
        model_paths=[checkpoint_path], threshold=best_threshold, debug=True
    )

    print(f"   Submission DataFrame Shape: {sub_df.shape}")
    print(f"   First few rows:\n{sub_df.head()}")

    submission_path = os.path.join(PathConfig.SUBMISSION_DIR, "submission.csv")
    assert os.path.exists(submission_path), "Submission file not found"

    # 8. Utility Verification (RLE)
    print("\n8. Verifying RLE Utilities...")
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE Decode does not match original mask"
    print("   RLE Encoding/Decoding Verified.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

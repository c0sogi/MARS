import os
import shutil
import numpy as np
import torch
import pandas as pd
import cv2
import warnings

# Import library components
from library.config import Config
from library.utils import (
    set_seed,
    rle_encode,
    rle_decode,
    pad_image,
    unpad_image,
    calculate_iou_batch,
    compute_map_score,
)
from library.dataset import SaltDataset, get_dataloader
from library.model import ResNeXt50UNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
from library.trainer import Trainer


def main():
    # =========================================================================
    # 1. Configuration & Setup
    # =========================================================================
    print(">>> Setting up configuration for demonstration...")

    # Override Config for speed and isolation
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    Config.SUBMISSION_PATH = os.path.join(Config.SUBMISSION_DIR, "submission.csv")

    # Update cache paths to point to the new temp directory
    Config.CACHE_TRAIN_IMAGES = os.path.join(Config.CACHE_DIR, "train_images.npy")
    Config.CACHE_TRAIN_MASKS = os.path.join(Config.CACHE_DIR, "train_masks.npy")
    Config.CACHE_TRAIN_DEPTHS = os.path.join(Config.CACHE_DIR, "train_depths.npy")
    Config.CACHE_VAL_IMAGES = os.path.join(Config.CACHE_DIR, "val_images.npy")
    Config.CACHE_VAL_MASKS = os.path.join(Config.CACHE_DIR, "val_masks.npy")
    Config.CACHE_VAL_DEPTHS = os.path.join(Config.CACHE_DIR, "val_depths.npy")
    Config.CACHE_TEST_IMAGES = os.path.join(Config.CACHE_DIR, "test_images.npy")
    Config.CACHE_TEST_DEPTHS = os.path.join(Config.CACHE_DIR, "test_depths.npy")
    Config.CACHE_TEST_IDS = os.path.join(Config.CACHE_DIR, "test_ids.npy")

    # Hyperparameters for fast demo
    Config.EPOCHS = 1
    Config.BATCH_SIZE = 8
    Config.NUM_WORKERS = 2
    Config.EARLY_STOPPING_PATIENCE = 1
    Config.SCHEDULER_PATIENCE = 1

    # Re-run setup to create new directories
    Config.setup()
    set_seed(Config.SEED)

    print(f"Working Directory: {Config.WORKING_DIR}")

    # =========================================================================
    # 2. Verify Utilities
    # =========================================================================
    print("\n>>> Verifying Utilities...")

    # Test RLE Encoding/Decoding
    mask_size = 101
    dummy_mask = np.zeros((mask_size, mask_size), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1  # Create a square of 1s

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(mask_size, mask_size))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(
        dummy_mask, decoded
    ), "RLE decode should reconstruct the original mask"
    print("RLE Encode/Decode: OK")

    # Test Padding/Unpadding
    dummy_img = np.random.randint(0, 255, (101, 101, 3), dtype=np.uint8)
    padded = pad_image(dummy_img, target_size=128)
    assert padded.shape == (128, 128, 3), f"Padded shape mismatch: {padded.shape}"

    unpadded = unpad_image(padded, original_size=101)
    assert unpadded.shape == (101, 101, 3), f"Unpadded shape mismatch: {unpadded.shape}"
    assert np.array_equal(
        dummy_img, unpadded
    ), "Unpadded image should match original (center crop check)"
    print("Image Padding/Unpadding: OK")

    # Test IoU Calculation
    pred_t = torch.tensor([[[1, 1, 0], [0, 0, 0]]]).float()
    target_t = torch.tensor([[[1, 0, 0], [0, 0, 0]]]).float()
    # Intersection = 1, Union = 2, IoU = 0.5
    iou = calculate_iou_batch(pred_t, target_t)
    assert torch.isclose(iou, torch.tensor([0.5])), f"IoU calculation failed. Got {iou}"
    print("IoU Calculation: OK")

    # =========================================================================
    # 3. Verify Dataset & DataLoader
    # =========================================================================
    print("\n>>> Verifying Dataset & DataLoader...")

    # Initialize Dataset
    train_ds = SaltDataset(mode="train")
    print(f"Train Dataset Size: {len(train_ds)}")

    # Get a sample
    sample_input, sample_mask, sample_id = train_ds[0]

    # Verify Shapes
    # Input: (3, 128, 128) -> [Seismic, Seismic, Depth]
    assert sample_input.shape == (
        3,
        128,
        128,
    ), f"Input tensor shape incorrect: {sample_input.shape}"
    # Mask: (128, 128)
    assert sample_mask.shape == (
        128,
        128,
    ), f"Mask tensor shape incorrect: {sample_mask.shape}"

    # Verify Channel Multiplexing
    # Channel 0 and 1 should be identical (Seismic image duplicated)
    assert torch.equal(
        sample_input[0], sample_input[1]
    ), "Channel 0 and 1 should be identical (Seismic)"
    # Channel 2 should be constant (Depth)
    depth_channel = sample_input[2]
    assert torch.min(depth_channel) == torch.max(
        depth_channel
    ), "Depth channel should be spatially constant"
    print("Dataset Shapes & Multiplexing: OK")

    # Verify DataLoader
    train_loader = get_dataloader("train", batch_size=4, shuffle=True)
    batch_imgs, batch_masks, batch_ids = next(iter(train_loader))
    assert batch_imgs.shape == (4, 3, 128, 128), "Batch image shape incorrect"
    assert batch_masks.shape == (4, 128, 128), "Batch mask shape incorrect"
    print("DataLoader: OK")

    # =========================================================================
    # 4. Verify Model
    # =========================================================================
    print("\n>>> Verifying Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNeXt50UNetPlusPlus(n_classes=1, deep_supervision=True).to(device)

    # Create dummy input
    dummy_input = torch.randn(2, 3, 128, 128).to(device)

    # Training Mode (Deep Supervision -> List of outputs)
    model.train()
    outputs = model(dummy_input)
    assert isinstance(
        outputs, list
    ), "Model in training mode should return a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs from Deep Supervision, got {len(outputs)}"
    assert outputs[-1].shape == (
        2,
        1,
        128,
        128,
    ), f"Final output shape mismatch: {outputs[-1].shape}"

    # Eval Mode (Inference -> Single output)
    model.eval()
    output = model(dummy_input)
    assert isinstance(output, torch.Tensor), "Model in eval mode should return a Tensor"
    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Inference output shape mismatch: {output.shape}"
    print("Model Architecture: OK")

    # =========================================================================
    # 5. Verify Losses
    # =========================================================================
    print("\n>>> Verifying Loss Functions...")

    bce_dice = BCEDiceLoss()
    lovasz = LovaszHingeLoss()

    # Dummy logits and targets
    logits = torch.randn(4, 1, 128, 128, requires_grad=True)
    targets = torch.randint(0, 2, (4, 128, 128)).float()

    # Test BCE+Dice
    loss_val = bce_dice(logits, targets)
    assert not torch.isnan(loss_val), "BCE+Dice returned NaN"
    loss_val.backward()  # Check gradient flow
    print("BCE+Dice Loss: OK")

    # Test Lovasz
    logits.grad = None  # Reset grad
    loss_val_lov = lovasz(logits, targets)
    assert not torch.isnan(loss_val_lov), "Lovasz returned NaN"
    loss_val_lov.backward()
    print("Lovasz Loss: OK")

    # =========================================================================
    # 6. Verify Trainer (Training Loop)
    # =========================================================================
    print("\n>>> Verifying Trainer & Training Loop...")

    trainer = Trainer()

    # To speed up, we can limit the number of batches in the loader by mocking,
    # but since we set EPOCHS=1 and BATCH_SIZE=8, it should be reasonably fast
    # (~300 iters). We'll let it run to prove robustness.

    print("Starting Training (1 Epoch)...")
    try:
        trainer.fit()
        print("Training Loop: OK")
    except Exception as e:
        print(f"Training Loop Failed: {e}")
        raise e

    # Check if checkpoint was created
    if os.path.exists(Config.BEST_MODEL_PATH):
        print(f"Checkpoint found at {Config.BEST_MODEL_PATH}")
    else:
        # If validation didn't improve (unlikely with random init vs real data, but possible),
        # force save for the next step
        print(
            "No best model saved (metrics might be low). Saving manually for submission test."
        )
        torch.save(
            {"model_state_dict": trainer.model.state_dict(), "best_threshold": 0.5},
            Config.BEST_MODEL_PATH,
        )

    # =========================================================================
    # 7. Verify Submission Generation
    # =========================================================================
    print("\n>>> Verifying Submission Generation...")

    trainer.generate_submission()

    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file created with {len(df)} rows.")
        assert len(df) > 0, "Submission file is empty"
        assert (
            "id" in df.columns and "rle_mask" in df.columns
        ), "Submission columns mismatch"
        print("Submission Generation: OK")
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n>>> All Demonstrations Completed Successfully.")


if __name__ == "__main__":
    main()

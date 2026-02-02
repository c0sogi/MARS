import os
import torch
import numpy as np
import shutil
import importlib
from torch.utils.data import DataLoader, Subset

# Import provided library modules
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, calculate_iou_map
from library.dataset import SaltDataset, get_transforms
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
import library.trainer

importlib.reload(library.trainer)
from library.trainer import SaltTrainer


def run_demo():
    print("=== Starting Salt Segmentation Demo ===\n")

    # -------------------------------------------------------------------------
    # 1. Configuration Setup for Speed
    # -------------------------------------------------------------------------
    print("[1] Configuring environment for rapid demonstration...")

    # Modify Config for a quick run
    Config.EPOCHS = 2
    Config.LOVASZ_EPOCH = 1  # Switch loss at epoch 1 to test both losses
    Config.BATCH_SIZE = 8  # Small batch size for demo
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_execution"
    Config.CACHE_DIR = os.path.join(Config.WORKING_DIR, "cache")
    Config.CHECKPOINT_DIR = os.path.join(Config.WORKING_DIR, "checkpoints")
    Config.SUBMISSION_DIR = os.path.join(Config.WORKING_DIR, "submission")
    Config.LOG_DIR = os.path.join(Config.WORKING_DIR, "logs")
    Config.BEST_MODEL_PATH = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    # Setup directories and seeds
    Config.setup()
    seed_everything(Config.SEED)
    print("    Configuration updated. Epochs: 2, Batch Size: 8.")

    # -------------------------------------------------------------------------
    # 2. Verify Utilities (RLE & IoU)
    # -------------------------------------------------------------------------
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    # Create a simple 101x101 mask with a 10x10 square of 1s
    dummy_mask = np.zeros((101, 101), dtype=np.uint8)
    dummy_mask[10:20, 10:20] = 1

    encoded = rle_encode(dummy_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    # Assertions
    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(dummy_mask, decoded), "Decoded mask must match original"
    print("    RLE Encoding/Decoding: PASSED")

    # Test IoU Calculation
    # Perfect match
    iou_perfect = calculate_iou_map(
        dummy_mask[None, ...], dummy_mask[None, ...], pixel_threshold=0.5
    )
    assert np.isclose(
        iou_perfect, 1.0
    ), f"Perfect match IoU should be 1.0, got {iou_perfect}"

    # No overlap
    empty_mask = np.zeros_like(dummy_mask)
    # Note: If both are empty, IoU is 1.0 (TN). If one is empty and other is not, IoU is 0.0.
    iou_miss = calculate_iou_map(
        dummy_mask[None, ...], empty_mask[None, ...], pixel_threshold=0.5
    )
    assert np.isclose(iou_miss, 0.0), f"No overlap IoU should be 0.0, got {iou_miss}"
    print("    IoU Calculation: PASSED")

    # -------------------------------------------------------------------------
    # 3. Verify Dataset & Transforms
    # -------------------------------------------------------------------------
    print("\n[3] Verifying Dataset and Transforms...")

    # Initialize dataset
    train_ds = SaltDataset(
        mode="train", transform=get_transforms(mode="train"), load_cached_data=False
    )

    # Fetch one sample
    img_tensor, mask_tensor, img_id = train_ds[0]

    # Assertions
    # Expected shape is (3, 128, 128) for image and (1, 128, 128) for mask
    # 128 is Config.IMG_SIZE
    assert img_tensor.shape == (
        3,
        128,
        128,
    ), f"Image shape mismatch: {img_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        128,
        128,
    ), f"Mask shape mismatch: {mask_tensor.shape}"
    assert isinstance(img_id, str), "Image ID should be a string"

    # Check normalization (approximate range)
    assert (
        img_tensor.min() >= -3.0 and img_tensor.max() <= 3.0
    ), "Image normalization seems off"
    assert (
        mask_tensor.min() >= 0.0 and mask_tensor.max() <= 1.0
    ), "Mask should be binary/float 0-1"

    print(f"    Dataset loaded. Sample ID: {img_id}")
    print(f"    Image Shape: {img_tensor.shape}, Mask Shape: {mask_tensor.shape}")
    print("    Dataset Verification: PASSED")

    # -------------------------------------------------------------------------
    # 4. Verify Model Architecture
    # -------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    device = Config.DEVICE
    model = SaltUNetPlusPlus().to(device)

    # Create dummy batch
    dummy_input = torch.randn(2, 3, 128, 128).to(device)

    # Test Training Mode (Deep Supervision)
    model.train()
    outputs_train = model(dummy_input)
    assert isinstance(
        outputs_train, list
    ), "Model in train mode should return list (Deep Supervision)"
    assert len(outputs_train) == 4, "Should return 4 outputs for deep supervision"
    assert outputs_train[-1].shape == (
        2,
        1,
        128,
        128,
    ), f"Output shape mismatch: {outputs_train[-1].shape}"

    # Test Eval Mode
    model.eval()
    with torch.no_grad():
        output_eval = model(dummy_input)
    assert isinstance(
        output_eval, torch.Tensor
    ), "Model in eval mode should return Tensor"
    assert output_eval.shape == (2, 1, 128, 128), "Eval output shape mismatch"

    print("    Model Forward Pass: PASSED")

    # -------------------------------------------------------------------------
    # 5. Verify Loss Functions
    # -------------------------------------------------------------------------
    print("\n[5] Verifying Loss Functions...")

    bce_dice = BCEDiceLoss().to(device)
    lovasz = LovaszHingeLoss().to(device)

    dummy_target = torch.randint(0, 2, (2, 1, 128, 128)).float().to(device)

    # Check BCE Dice with Deep Supervision input
    loss_val_bce = bce_dice(outputs_train, dummy_target)
    assert not torch.isnan(loss_val_bce), "BCE Dice loss returned NaN"
    assert loss_val_bce > 0, "BCE Dice loss should be positive"

    # Check Lovasz with Deep Supervision input (should pick last output)
    loss_val_lovasz = lovasz(outputs_train, dummy_target)
    assert not torch.isnan(loss_val_lovasz), "Lovasz loss returned NaN"

    print("    Loss Calculation: PASSED")

    # -------------------------------------------------------------------------
    # 6. Verify Training Loop (Trainer)
    # -------------------------------------------------------------------------
    print("\n[6] Running Training Loop Demo...")

    trainer = SaltTrainer()

    # OPTIMIZATION: Monkey-patch dataloaders to use a tiny subset for speed
    # We select 16 indices for train and 8 for val
    train_indices = list(range(16))
    val_indices = list(range(8))

    trainer.train_loader = DataLoader(
        Subset(trainer.train_loader.dataset, train_indices),
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    trainer.val_loader = DataLoader(
        Subset(trainer.val_loader.dataset, val_indices),
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print(
        f"    Subset active: {len(trainer.train_loader)} train batches, {len(trainer.val_loader)} val batches."
    )

    # Run training
    trainer.fit()

    # Verify artifact generation
    assert os.path.exists(
        Config.BEST_MODEL_PATH
    ), "Best model checkpoint was not created!"
    print(f"    Checkpoint created at: {Config.BEST_MODEL_PATH}")
    print("    Training Loop: PASSED")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

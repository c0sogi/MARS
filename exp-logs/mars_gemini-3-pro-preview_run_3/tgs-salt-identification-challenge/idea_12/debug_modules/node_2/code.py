import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import cv2
import time

# Import from the provided library
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, do_kaggle_metric
from library.dataset import SaltDataset, get_transforms, get_fold_loaders
from library.model import UNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss
import importlib
import library.train

importlib.reload(library.train)
from library.train import train_fold


def demo_utils():
    print("\n=== Demo: Utilities (RLE & Metric) ===")

    # 1. Test RLE Encode/Decode
    # Create a simple 101x101 mask with a 10x10 square of 1s
    mask = np.zeros((101, 101), dtype=np.uint8)
    mask[10:20, 10:20] = 1

    encoded = rle_encode(mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert isinstance(encoded, str), "RLE encode should return a string"
    assert np.array_equal(mask, decoded), "Decoded mask does not match original"
    print("RLE Encode/Decode logic verified.")

    # 2. Test Kaggle Metric (mAP at IoU thresholds)
    # Case 1: Perfect match
    score_perfect = do_kaggle_metric(mask[None, ...], mask[None, ...], threshold=0.5)
    assert np.isclose(
        score_perfect, 1.0
    ), f"Perfect match should have score 1.0, got {score_perfect}"

    # Case 2: No overlap
    mask_inv = np.zeros_like(mask)
    mask_inv[50:60, 50:60] = 1
    score_zero = do_kaggle_metric(mask[None, ...], mask_inv[None, ...], threshold=0.5)
    assert np.isclose(
        score_zero, 0.0
    ), f"No overlap should have score 0.0, got {score_zero}"

    print("Kaggle Metric logic verified.")


def demo_dataset():
    print("\n=== Demo: Dataset & Transforms ===")

    # Create dummy data simulating the inputs
    # 10 samples, 101x101 images
    num_samples = 10
    images = np.random.randint(0, 255, (num_samples, 101, 101), dtype=np.uint8)
    depths = np.random.rand(num_samples).astype(np.float32)  # Normalized depths
    masks = np.random.randint(0, 2, (num_samples, 101, 101), dtype=np.uint8)
    ids = [f"id_{i}" for i in range(num_samples)]

    # Instantiate Dataset with Train transforms
    ds_train = SaltDataset(
        images=images,
        depths=depths,
        masks=masks,
        ids=ids,
        phase="train",
        transforms=get_transforms("train"),
    )

    # Fetch one sample
    input_tensor, mask_tensor = ds_train[0]

    # Verify Shapes
    # Config.IMG_SIZE is 128. Transforms should pad 101 -> 128.
    # Input channels: 3 (Seismic, Seismic, Depth)
    print(f"Input Tensor Shape: {input_tensor.shape}")
    print(f"Mask Tensor Shape: {mask_tensor.shape}")

    assert input_tensor.shape == (
        3,
        128,
        128,
    ), f"Expected (3, 128, 128), got {input_tensor.shape}"
    assert mask_tensor.shape == (
        1,
        128,
        128,
    ), f"Expected (1, 128, 128), got {mask_tensor.shape}"

    # Verify Data Range
    assert (
        input_tensor.min() >= 0.0 and input_tensor.max() <= 1.0
    ), "Input tensor not normalized [0, 1]"

    print("Dataset processing and augmentation verified.")


def demo_model():
    print("\n=== Demo: UNet++ Model Architecture ===")

    device = Config.DEVICE
    model = UNetPlusPlus().to(device)

    # Create dummy batch: Batch Size 2, 3 Channels, 128x128
    x = torch.randn(2, 3, 128, 128).to(device)

    # 1. Test Training Mode (Deep Supervision)
    model.train()
    outputs = model(x)

    assert isinstance(
        outputs, list
    ), "Model in train mode should return a list (Deep Supervision)"
    assert (
        len(outputs) == 4
    ), f"Expected 4 outputs for deep supervision, got {len(outputs)}"
    for i, out in enumerate(outputs):
        assert out.shape == (2, 1, 128, 128), f"Output {i} shape mismatch: {out.shape}"
    print("Model Training forward pass verified.")

    # 2. Test Eval Mode (Single Output)
    model.eval()
    with torch.no_grad():
        output = model(x)

    assert torch.is_tensor(output), "Model in eval mode should return a single tensor"
    assert output.shape == (
        2,
        1,
        128,
        128,
    ), f"Eval output shape mismatch: {output.shape}"
    print("Model Evaluation forward pass verified.")

    # Clean up
    del model, x, outputs, output
    torch.cuda.empty_cache()


def demo_losses():
    print("\n=== Demo: Loss Functions ===")

    device = Config.DEVICE
    b_size = 4

    # Dummy logits (raw model output) and targets (binary 0/1)
    logits = torch.randn(b_size, 1, 128, 128).to(device)
    targets = torch.randint(0, 2, (b_size, 1, 128, 128)).float().to(device)

    # 1. BCE Dice Loss
    criterion_bce = BCEDiceLoss()
    loss_bce = criterion_bce(logits, targets)

    assert not torch.isnan(loss_bce), "BCE Dice Loss returned NaN"
    assert loss_bce.item() > 0, "BCE Dice Loss should be positive"
    print(f"BCE Dice Loss verified. Value: {loss_bce.item():.4f}")

    # 2. Lovasz Hinge Loss
    criterion_lovasz = LovaszHingeLoss()
    loss_lovasz = criterion_lovasz(logits, targets)

    assert not torch.isnan(loss_lovasz), "Lovasz Loss returned NaN"
    print(f"Lovasz Hinge Loss verified. Value: {loss_lovasz.item():.4f}")

    # Clean up
    del logits, targets, loss_bce, loss_lovasz
    torch.cuda.empty_cache()


def demo_training_loop():
    print("\n=== Demo: Training Loop (Fold 0, Debug Mode) ===")

    # Ensure metadata exists (it should based on problem description)
    if not os.path.exists(Config.TRAIN_METADATA):
        raise FileNotFoundError(f"Metadata not found at {Config.TRAIN_METADATA}")

    print("Starting training for Fold 0 with debug=True (2 epochs)...")

    # Run training
    # This uses the real data provided in ./input and ./metadata
    # debug=True limits epochs to Config.DEBUG_EPOCHS (2)
    try:
        best_score = train_fold(fold_idx=0, debug=True)

        print(f"Training demo completed successfully.")
        print(f"Best Validation mAP Score: {best_score:.4f}")

        # Verify checkpoint creation
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "fold_0_best.pth")
        assert os.path.exists(
            checkpoint_path
        ), f"Checkpoint not found at {checkpoint_path}"
        print("Checkpoint saved successfully.")

    except Exception as e:
        print(f"Training loop failed with error: {e}")
        raise e


if __name__ == "__main__":
    # Set global seed
    seed_everything(Config.SEED)

    # Run demonstrations
    demo_utils()
    demo_dataset()
    demo_model()
    demo_losses()

    # Run integration test (Training Loop)
    # This might take a few minutes depending on dataset size, but debug mode keeps it short.
    demo_training_loop()

    print("\nAll demonstrations completed successfully.")

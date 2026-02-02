import sys
import os
import torch
import numpy as np
import warnings
import cv2

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import library modules
from library.config import Config
from library.utils import seed_everything, rle_encode, rle_decode, iou_metric, calc_map
from library.dataset import get_loaders
from library.model import SaltUNetPlusPlus
from library.losses import BCEDiceLoss, LovaszHingeLoss, DeepSupervisionLoss
from library.engine import train_one_epoch, evaluate


def run_demo():
    print("=== Salt Segmentation Pipeline Demo ===")

    # 1. Setup Configuration for Speed and Demo constraints
    print("\n[1] Configuring environment...")
    Config.DEBUG = True
    Config.DEBUG_SIZE = 16  # Use a tiny subset of data
    Config.BATCH_SIZE = 4  # Small batch size
    Config.NUM_WORKERS = 0  # Disable multiprocessing for simple script execution
    Config.ENCODER_NAME = "resnet18"  # Use lightweight encoder for speed
    Config.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Ensure reproducibility
    seed_everything(Config.SEED)
    print(f"    Device: {Config.DEVICE}")
    print(f"    Encoder: {Config.ENCODER_NAME}")

    # 2. Verify Utility Functions
    print("\n[2] Verifying Utility Functions...")

    # Test RLE Encoding/Decoding
    # Create a synthetic 101x101 mask with a 10x10 square of salt
    gt_mask = np.zeros((101, 101), dtype=np.uint8)
    gt_mask[10:20, 10:20] = 1

    encoded = rle_encode(gt_mask)
    decoded = rle_decode(encoded, shape=(101, 101))

    assert np.array_equal(gt_mask, decoded), "RLE Decode does not match original mask!"
    print("    RLE Encode/Decode: PASSED")

    # Test IoU Metric
    # Perfect overlap
    iou_perfect = iou_metric(gt_mask, gt_mask)
    assert iou_perfect == 1.0, f"Expected IoU 1.0, got {iou_perfect}"

    # No overlap
    empty_mask = np.zeros_like(gt_mask)
    iou_zero = iou_metric(gt_mask, empty_mask)
    assert iou_zero == 0.0, f"Expected IoU 0.0, got {iou_zero}"
    print("    IoU Metric Logic:  PASSED")

    # 3. Verify Data Loading
    print("\n[3] Verifying Data Loading...")
    # Force reload to ignore any potential existing caches from different runs
    train_loader, val_loader = get_loaders(fold=0, debug=True, load_cached_data=False)

    # Fetch one batch
    images, masks, ids = next(iter(train_loader))

    # Verify Shapes
    # Expected: (B, 3, 128, 128) -> 3 channels: Seismic, Seismic, Depth
    # Expected: (B, 1, 128, 128) -> Mask padded to model size
    assert images.shape == (
        Config.BATCH_SIZE,
        3,
        128,
        128,
    ), f"Image shape mismatch: {images.shape}"
    assert masks.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Mask shape mismatch: {masks.shape}"

    # Verify Value Ranges
    assert (
        images.min() >= 0.0 and images.max() <= 1.0
    ), "Images should be normalized to [0, 1]"
    assert set(np.unique(masks.numpy())).issubset(
        {0, 1}
    ), "Masks should be binary (0, 1)"

    print(f"    Batch Images: {images.shape}")
    print(f"    Batch Masks:  {masks.shape}")
    print("    Data Loader:  PASSED")

    # 4. Verify Model Architecture
    print("\n[4] Verifying Model Architecture...")
    model = SaltUNetPlusPlus()
    model.to(Config.DEVICE)

    # Perform Forward Pass
    images = images.to(Config.DEVICE)
    outputs = model(images)

    # Check Deep Supervision Output
    assert isinstance(
        outputs, list
    ), "Model should return a list of outputs for deep supervision"
    assert len(outputs) == 4, f"Expected 4 output heads, got {len(outputs)}"

    # Check shape of final output
    final_out = outputs[-1]
    assert final_out.shape == (
        Config.BATCH_SIZE,
        1,
        128,
        128,
    ), f"Output shape mismatch: {final_out.shape}"

    print(f"    Output Heads: {len(outputs)}")
    print(f"    Final Shape:  {final_out.shape}")
    print("    Model Forward: PASSED")

    # 5. Verify Loss Functions
    print("\n[5] Verifying Loss Functions...")
    masks = masks.to(Config.DEVICE)

    # Setup Loss: BCE + Dice with Deep Supervision
    # Weights for 4 heads: [0.1, 0.1, 0.1, 1.0]
    base_loss = BCEDiceLoss()
    criterion = DeepSupervisionLoss(base_loss, weights=[0.1, 0.1, 0.1, 1.0])

    loss = criterion(outputs, masks)

    assert torch.is_tensor(loss), "Loss should be a tensor"
    assert loss.ndim == 0, "Loss should be a scalar"
    assert not torch.isnan(loss), "Loss is NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"    Calculated Loss: {loss.item():.6f}")
    print("    Loss Function:   PASSED")

    # 6. Verify Training Engine
    print("\n[6] Verifying Training & Evaluation Loop...")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(Config.DEVICE == "cuda"))

    # Run 1 Training Epoch
    print("    Running training epoch...")
    train_loss = train_one_epoch(
        model,
        train_loader,
        optimizer,
        scaler,
        criterion,
        Config.DEVICE,
        deep_supervision=True,
    )
    print(f"    > Train Loss: {train_loss:.4f}")

    # Run Evaluation
    # Use Lovasz Hinge for validation metric
    print("    Running evaluation...")
    val_criterion = LovaszHingeLoss()
    val_loss, val_map = evaluate(
        model, val_loader, val_criterion, Config.DEVICE, deep_supervision=False
    )

    print(f"    > Val Loss: {val_loss:.4f}")
    print(f"    > Val mAP:  {val_map:.4f}")

    assert train_loss >= 0, "Training loss invalid"
    assert val_map >= 0 and val_map <= 1.0, "mAP score out of range"

    print("    Engine: PASSED")

    print("\n=== All Demonstrations Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

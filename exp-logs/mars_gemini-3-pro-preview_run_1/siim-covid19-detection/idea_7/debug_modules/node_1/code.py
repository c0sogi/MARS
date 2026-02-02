import os
import sys
import torch
import torch.optim as optim
import numpy as np
import warnings

# Add the current directory to path to ensure library imports work
sys.path.append(os.getcwd())

# Import from provided library files
from library.config import Config, seed_everything
from library.dataset import get_loaders
from library.model import DeepSupervisedResNet18UNet
from library.engine import train_one_epoch, valid_one_epoch
from library.utils import get_bbox_from_mask, format_prediction_string

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demonstration():
    print("Starting SIIM-COVID19 Detection Pipeline Demonstration...")

    # 1. Setup & Configuration
    # ------------------------
    print("\n[1] Setting up environment...")
    seed_everything(Config.SEED)

    # Override Config for quick demonstration
    img_size = 256  # Smaller size for speed
    batch_size = 4
    debug_samples = 16
    device = Config.DEVICE

    print(f"    Device: {device}")
    print(f"    Image Size: {img_size}")
    print(f"    Batch Size: {batch_size}")
    print(f"    Debug Mode: True (Samples: {debug_samples})")

    # 2. Data Loading
    # ---------------
    print("\n[2] Initializing DataLoaders...")

    # We use num_workers=0 to avoid multiprocessing overhead in this short script
    train_loader, val_loader, test_loader = get_loaders(
        train_csv_path=Config.TRAIN_CSV,
        val_csv_path=Config.VAL_CSV,
        test_csv_path=Config.TEST_CSV,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=0,
        debug=True,
        debug_sample_size=debug_samples,
        load_cached_data=False,  # Force processing to demonstrate pipeline
    )

    # Fetch a single batch to verify data structure
    images, masks, labels = next(iter(train_loader))

    print(f"    Fetched batch shapes:")
    print(
        f"    - Images: {images.shape} (Expected: [{batch_size}, 3, {img_size}, {img_size}])"
    )
    print(
        f"    - Masks : {masks.shape}  (Expected: [{batch_size}, 1, {img_size}, {img_size}])"
    )
    print(f"    - Labels: {labels.shape} (Expected: [{batch_size}, 4])")

    # Assertions
    assert images.shape == (
        batch_size,
        3,
        img_size,
        img_size,
    ), "Image batch shape mismatch"
    assert masks.shape == (
        batch_size,
        1,
        img_size,
        img_size,
    ), "Mask batch shape mismatch"
    assert labels.shape == (batch_size, 4), "Label batch shape mismatch"
    assert images.dtype == torch.float32, "Images should be float32"
    assert masks.dtype == torch.float32, "Masks should be float32"

    print("    Data Loading verification passed.")

    # 3. Model Initialization & Forward Pass
    # --------------------------------------
    print("\n[3] Initializing Model...")
    model = DeepSupervisedResNet18UNet(
        pretrained=False
    )  # False for speed, we just check logic
    model = model.to(device)

    print("    Performing forward pass...")
    images = images.to(device)

    # Forward pass
    logit_cls, logit_seg_final, logit_seg_aux1, logit_seg_aux2 = model(images)

    print(f"    Output shapes:")
    print(f"    - Class Logits: {logit_cls.shape}")
    print(f"    - Seg Final   : {logit_seg_final.shape}")
    print(f"    - Seg Aux1    : {logit_seg_aux1.shape}")
    print(f"    - Seg Aux2    : {logit_seg_aux2.shape}")

    # Assertions
    assert logit_cls.shape == (batch_size, 4), "Classification output shape mismatch"
    assert logit_seg_final.shape == (
        batch_size,
        1,
        img_size,
        img_size,
    ), "Final segmentation shape mismatch"
    assert logit_seg_aux1.shape == (
        batch_size,
        1,
        img_size // 2,
        img_size // 2,
    ), "Aux1 segmentation shape mismatch"
    assert logit_seg_aux2.shape == (
        batch_size,
        1,
        img_size // 4,
        img_size // 4,
    ), "Aux2 segmentation shape mismatch"

    print("    Model forward pass verification passed.")

    # 4. Training Loop Simulation
    # ---------------------------
    print("\n[4] Simulating Training Epoch...")

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    # Run one epoch
    train_loss = train_one_epoch(
        model, optimizer, scheduler, train_loader, device, epoch=1
    )

    print(f"    Training Loss: {train_loss:.4f}")
    assert np.isfinite(train_loss), "Training loss is NaN or Infinite"
    print("    Training step verification passed.")

    # 5. Validation Loop Simulation
    # -----------------------------
    print("\n[5] Simulating Validation Epoch...")

    # Run validation
    metrics = valid_one_epoch(model, val_loader, device, epoch=1)

    print("    Validation Metrics:")
    for k, v in metrics.items():
        print(f"    - {k}: {v:.4f}")

    assert "loss" in metrics and np.isfinite(metrics["loss"]), "Validation loss invalid"
    assert (
        "study_acc" in metrics and 0 <= metrics["study_acc"] <= 1
    ), "Study accuracy out of range"
    assert (
        "map_score" in metrics and 0 <= metrics["map_score"] <= 1
    ), "mAP score out of range"

    print("    Validation step verification passed.")

    # 6. Inference & Post-processing Logic
    # ------------------------------------
    print("\n[6] Demonstrating Inference Post-processing...")

    # Simulate a prediction mask (binary square in the middle)
    dummy_mask = np.zeros((100, 100), dtype=np.uint8)
    dummy_mask[20:50, 20:50] = 1  # 30x30 square

    # Extract boxes
    boxes = get_bbox_from_mask(dummy_mask)
    print(f"    Extracted Boxes from dummy mask: {boxes}")

    # Verify box coordinates [x1, y1, x2, y2]
    # cv2.boundingRect returns x, y, w, h. get_bbox_from_mask converts to x1, y1, x2, y2
    # Expected: [20, 20, 50, 50] (approximate depending on contour finding)
    assert len(boxes) == 1, "Should detect exactly one box"
    b = boxes[0]
    # Allow 1px tolerance for contour approximation
    assert 19 <= b[0] <= 21 and 19 <= b[1] <= 21, "Box start coordinates incorrect"
    assert 49 <= b[2] <= 51 and 49 <= b[3] <= 51, "Box end coordinates incorrect"

    # Format Prediction String
    pred_str = format_prediction_string("opacity", 0.95, boxes)
    print(f"    Formatted Prediction String: '{pred_str}'")

    # Verify string format
    parts = pred_str.split()
    assert parts[0] == "opacity", "Label incorrect"
    assert parts[1] == "0.9500", "Confidence incorrect"
    assert len(parts) == 6, "String should have 6 parts for 1 box"

    # Test "none" case
    none_str = format_prediction_string("none", 1.0, [])
    print(f"    Formatted 'None' String: '{none_str}'")
    assert none_str == "none 1 0 0 1 1", "None format incorrect"

    print("    Post-processing verification passed.")

    print("\nAll demonstrations completed successfully!")


if __name__ == "__main__":
    run_demonstration()

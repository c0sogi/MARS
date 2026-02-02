import os
import sys
import warnings
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader

# Import provided library modules
from library.config import Config
from library.utils import set_seed, rle_encode, dice_coef_metric
from library.dataset import ContrailDataset
from library.model import MultiTaskResNetUNet
from library.loss import MultiTaskLoss
from library.train import train_one_epoch, validate

# Suppress warnings for clean output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Contrail Detection Library Demo ===\n")

    # ------------------------------------------------------------------------
    # 1. Configuration Setup
    # ------------------------------------------------------------------------
    print("[1] Configuring environment for fast demonstration...")
    # Override Config for speed
    Config.DEBUG = True
    Config.BATCH_SIZE = 2
    Config.EPOCHS = 1
    Config.NUM_WORKERS = 0  # Avoid multiprocessing overhead for small demo
    Config.WORKING_DIR = "./working/demo_run"
    Config.OUTPUT_DIR = "./working/demo_submission"
    Config.SUBMISSION_FILE = os.path.join(Config.OUTPUT_DIR, "submission.csv")

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"    Device: {device}")
    print("    Configuration updated for demo.")

    # ------------------------------------------------------------------------
    # 2. Utility Verification
    # ------------------------------------------------------------------------
    print("\n[2] Verifying Utilities...")

    # Test set_seed
    set_seed(42)
    rand_val_1 = np.random.rand()
    set_seed(42)
    rand_val_2 = np.random.rand()
    assert rand_val_1 == rand_val_2, "set_seed failed to ensure reproducibility."
    print("    set_seed: Passed")

    # Test rle_encode
    # Create a 2x2 mask: [[0, 1], [0, 0]]
    # Flattened F-order (col-major): [0 (0,0), 0 (1,0), 1 (0,1), 0 (1,1)]
    # Indices (1-based): 1, 2, 3, 4. Value 1 is at index 3.
    # Expected RLE: "3 1"
    dummy_mask = np.array([[0, 1], [0, 0]], dtype=np.uint8)
    rle_out = rle_encode(dummy_mask)
    assert rle_out == "3 1", f"rle_encode failed. Expected '3 1', got '{rle_out}'"
    print("    rle_encode: Passed")

    # Test dice_coef_metric
    # Pred: [1, 1], True: [1, 0]
    # Intersection: 1. Union: 2 + 1 = 3. Dice: 2*1/3 = 0.666...
    y_pred = torch.tensor([1.0, 1.0])
    y_true = torch.tensor([1.0, 0.0])
    dice_val = dice_coef_metric(y_pred, y_true, threshold=0.5)
    assert abs(dice_val - 0.666666) < 1e-4, f"dice_coef_metric failed. Got {dice_val}"
    print("    dice_coef_metric: Passed")

    # ------------------------------------------------------------------------
    # 3. Dataset Verification
    # ------------------------------------------------------------------------
    print("\n[3] Verifying Dataset...")

    # Initialize dataset with max_samples constraint
    train_ds = ContrailDataset(split="train", max_samples=10)
    print(f"    Loaded Train Dataset with {len(train_ds)} samples.")

    # Fetch one sample
    image, mask, label, record_id = train_ds[0]

    # Verify shapes
    # Image: (6, 256, 256), Mask: (1, 256, 256), Label: scalar
    assert image.shape == (6, 256, 256), f"Image shape mismatch: {image.shape}"
    assert mask.shape == (1, 256, 256), f"Mask shape mismatch: {mask.shape}"
    assert isinstance(record_id, str), "Record ID should be a string"
    assert label.numel() == 1, "Label should be scalar"

    # Verify normalization (approximate range check)
    assert (
        image.min() >= 0.0 and image.max() <= 1.0
    ), "Image data not normalized to [0, 1]"

    print("    Dataset shapes and types: Passed")

    # ------------------------------------------------------------------------
    # 4. Model Verification
    # ------------------------------------------------------------------------
    print("\n[4] Verifying Model Architecture...")

    model = MultiTaskResNetUNet(in_channels=Config.IN_CHANNELS, pretrained=False)
    model.to(device)

    # Create dummy batch: (B, C, H, W)
    dummy_input = torch.randn(2, 6, 256, 256).to(device)

    # Forward pass
    seg_logits, cls_logits = model(dummy_input)

    # Verify output shapes
    # Seg: (B, 1, 256, 256), Cls: (B, 1)
    assert seg_logits.shape == (
        2,
        1,
        256,
        256,
    ), f"Seg logits shape mismatch: {seg_logits.shape}"
    assert cls_logits.shape == (2, 1), f"Cls logits shape mismatch: {cls_logits.shape}"

    print("    Model forward pass and output shapes: Passed")

    # ------------------------------------------------------------------------
    # 5. Loss Verification
    # ------------------------------------------------------------------------
    print("\n[5] Verifying Loss Function...")

    criterion = MultiTaskLoss()

    # Dummy targets
    dummy_mask_target = torch.zeros(2, 1, 256, 256).to(device)
    dummy_cls_target = torch.zeros(2, 1).to(device)

    loss_dict = criterion(seg_logits, cls_logits, dummy_mask_target, dummy_cls_target)

    assert "loss" in loss_dict, "Total loss missing in loss dict"
    assert "seg_loss" in loss_dict, "Segmentation loss missing"
    assert "cls_loss" in loss_dict, "Classification loss missing"

    # Verify backward pass capability
    loss_val = loss_dict["loss"]
    loss_val.backward()
    print("    Loss calculation and backward pass: Passed")

    # ------------------------------------------------------------------------
    # 6. Training Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[6] Demonstrating Training Loop (1 Epoch)...")

    # Re-initialize model and optimizer
    model = MultiTaskResNetUNet(in_channels=Config.IN_CHANNELS, pretrained=False).to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Create DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
    )

    # Capture initial weights of a specific layer to verify update
    initial_weight = model.seg_head.weight.clone()

    # Run one epoch
    epoch_loss, epoch_seg, epoch_cls = train_one_epoch(
        model, train_loader, optimizer, criterion, device
    )

    print(
        f"    Epoch Loss: {epoch_loss:.4f} (Seg: {epoch_seg:.4f}, Cls: {epoch_cls:.4f})"
    )

    # Verify weights updated
    final_weight = model.seg_head.weight
    assert not torch.equal(
        initial_weight, final_weight
    ), "Model weights did not update!"
    print("    Model weights updated successfully: Passed")

    # ------------------------------------------------------------------------
    # 7. Validation Loop Demonstration
    # ------------------------------------------------------------------------
    print("\n[7] Demonstrating Validation Loop...")

    val_ds = ContrailDataset(split="validation", max_samples=10)
    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    val_loss, val_dice, val_acc = validate(model, val_loader, criterion, device)

    print(f"    Val Loss: {val_loss:.4f}")
    print(f"    Val Dice: {val_dice:.4f}")
    print(f"    Val Acc:  {val_acc:.4f}")

    assert isinstance(val_dice, float), "Validation Dice should be a float"
    print("    Validation metrics computed: Passed")

    # ------------------------------------------------------------------------
    # 8. Inference & Submission Demonstration
    # ------------------------------------------------------------------------
    print("\n[8] Demonstrating Inference and Submission Generation...")

    # Use Test Dataset
    test_ds = ContrailDataset(split="test", max_samples=10)
    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    model.eval()
    submission_data = []

    with torch.no_grad():
        for images, _, _, record_ids in test_loader:
            images = images.to(device)

            # Forward
            seg_logits, cls_logits = model(images)

            # Gated Inference Logic
            seg_probs = torch.sigmoid(seg_logits)
            cls_probs = torch.sigmoid(cls_logits)

            # Thresholding
            pred_masks = (seg_probs > Config.PIXEL_THRESHOLD).float()
            gate_mask = (cls_probs > Config.CLS_THRESHOLD).float().view(-1, 1, 1, 1)
            pred_masks = pred_masks * gate_mask

            # Encode
            pred_masks_np = pred_masks.squeeze(1).cpu().numpy().astype(np.uint8)

            for i, record_id in enumerate(record_ids):
                mask = pred_masks_np[i]
                rle = rle_encode(mask)
                submission_data.append({"record_id": record_id, "encoded_pixels": rle})

    # Save submission
    df_sub = pd.DataFrame(submission_data)
    df_sub.to_csv(Config.SUBMISSION_FILE, index=False)

    print(f"    Submission generated with {len(df_sub)} records.")
    print(f"    File saved to: {Config.SUBMISSION_FILE}")

    # Verify file exists and format
    assert os.path.exists(Config.SUBMISSION_FILE), "Submission file was not created"
    df_check = pd.read_csv(Config.SUBMISSION_FILE)
    assert (
        "record_id" in df_check.columns and "encoded_pixels" in df_check.columns
    ), "Submission columns mismatch"
    print("    Submission format verification: Passed")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

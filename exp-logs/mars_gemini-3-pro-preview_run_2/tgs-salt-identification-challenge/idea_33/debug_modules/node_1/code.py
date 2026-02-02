import os
import sys
import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import cv2
import warnings

# Import from the provided library files
from library.utils import seed_everything, rle_encode, pad_image, unpad_image, calc_map
from library.losses import SaltLoss
from library.model import ResNet34WideLinkNet
from library.dataset import load_data_from_metadata, SaltDataset, get_transforms
from library.engine import train_one_epoch, evaluate, predict_marginalized

# Suppress warnings
warnings.filterwarnings("ignore")


def test_utils():
    """Verifies utility functions: padding, unpadding, RLE encoding, and mAP calculation."""
    print("[1/5] Testing Utility Functions...")

    # 1. Test Padding and Unpadding
    original_size = 101
    target_size = 128
    dummy_img = np.random.randint(
        0, 255, (original_size, original_size), dtype=np.uint8
    )

    padded_img = pad_image(dummy_img, target_size=target_size)
    assert padded_img.shape == (
        target_size,
        target_size,
    ), f"Padding failed. Expected ({target_size}, {target_size}), got {padded_img.shape}"

    unpadded_img = unpad_image(padded_img, original_size=original_size)
    assert unpadded_img.shape == (
        original_size,
        original_size,
    ), f"Unpadding failed. Expected ({original_size}, {original_size}), got {unpadded_img.shape}"

    # Check content preservation (center crop of reflection padding should match original)
    # Note: Reflection padding might make edges tricky, but unpad simply crops the center.
    # Since pad_image centers the image, unpad_image should retrieve it exactly.
    np.testing.assert_array_equal(
        dummy_img, unpadded_img, err_msg="Unpadded image does not match original."
    )

    # 2. Test RLE Encoding
    # Create a simple mask: 3x3, pixels (0,0) and (0,1) are 1.
    # Flattened (column-major): 1, 0, 0, 1, 0, 0, 0, 0, 0 -> Indices 1 and 4?
    # Wait, RLE is column-major.
    # 3x3 Matrix:
    # 1 0 0
    # 1 0 0
    # 0 0 0
    # Flattened: 1, 1, 0, 0, 0, 0, 0, 0, 0. Indices: 1, 2. Run: 1 2.
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[0, 0] = 1
    mask[1, 0] = 1
    rle = rle_encode(mask)
    assert rle == "1 2", f"RLE Encoding incorrect. Expected '1 2', got '{rle}'"

    # 3. Test mAP Calculation
    # Case 1: Perfect match
    preds = np.zeros((2, 101, 101), dtype=np.uint8)
    targets = np.zeros((2, 101, 101), dtype=np.uint8)
    preds[0, 10:20, 10:20] = 1
    targets[0, 10:20, 10:20] = 1

    score = calc_map(preds, targets)
    assert score == 1.0, f"mAP calculation failed for perfect match. Got {score}"

    # Case 2: No overlap
    preds[1, 50:60, 50:60] = 1
    targets[1, 70:80, 70:80] = 1
    score = calc_map(preds[1:], targets[1:])  # Check single image
    assert score == 0.0, f"mAP calculation failed for no overlap. Got {score}"

    print(" - Utils verified successfully.")


def test_model_logic():
    """Verifies model instantiation and forward pass dimensions."""
    print("[2/5] Testing Model Architecture...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResNet34WideLinkNet(pretrained=False).to(device)

    batch_size = 4
    # Input: (B, 1, 128, 128)
    dummy_input = torch.randn(batch_size, 1, 128, 128).to(device)
    # Depth: (B)
    dummy_depth = torch.randn(batch_size).to(device)

    output = model(dummy_input, dummy_depth)

    expected_shape = (batch_size, 1, 128, 128)
    assert (
        output.shape == expected_shape
    ), f"Model output shape incorrect. Expected {expected_shape}, got {output.shape}"

    print(" - Model forward pass verified successfully.")
    return model, device


def test_loss_function(device):
    """Verifies the custom SaltLoss (BCE + Lovasz)."""
    print("[3/5] Testing Loss Function...")

    criterion = SaltLoss(bce_weight=1.0, lovasz_weight=1.0)

    # Logits: (B, 1, 128, 128)
    logits = torch.randn(4, 1, 128, 128, device=device, requires_grad=True)
    # Targets: (B, 1, 128, 128) Binary
    targets = torch.randint(0, 2, (4, 1, 128, 128)).float().to(device)

    loss = criterion(logits, targets)

    assert torch.is_tensor(loss), "Loss is not a tensor."
    assert loss.dim() == 0, "Loss should be a scalar."
    assert not torch.isnan(loss), "Loss returned NaN."

    # Verify backward pass works
    loss.backward()
    assert logits.grad is not None, "Gradients not computed."

    print(" - Loss function verified successfully.")


def test_training_and_inference_pipeline(model, device):
    """
    Demonstrates the full pipeline using a data subset:
    1. Load Data (Subset)
    2. Create Dataset/Loader
    3. Run Training Step
    4. Run Inference Step
    """
    print("[4/5] Testing Training & Inference Pipeline (Subset)...")

    # 1. Load Data manually to create a small subset
    train_meta_path = "./metadata/train.csv"
    # This function caches data to ./working/idea_33/
    images, masks, depths, ids = load_data_from_metadata(
        train_meta_path, load_cached_data=True, cache_name="train"
    )

    # Subset to 16 images for speed
    subset_idx = np.arange(16)
    sub_images = images[subset_idx]
    sub_masks = masks[subset_idx]
    sub_depths = depths[subset_idx]
    sub_ids = ids[subset_idx]

    # Calculate stats
    d_mean = np.mean(sub_depths)
    d_std = np.std(sub_depths)
    depth_stats = (d_mean, d_std)

    # Create Dataset
    train_dataset = SaltDataset(
        sub_images,
        sub_masks,
        sub_depths,
        sub_ids,
        transform=get_transforms("train"),
        depth_stats=depth_stats,
    )

    # Create Loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=4, shuffle=True, num_workers=0, drop_last=True
    )

    # Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = SaltLoss()

    # 2. Run one epoch of training
    print(" - Running 1 epoch on 16 samples...")
    epoch_loss = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch=0
    )
    print(f"   Train Loss: {epoch_loss:.4f}")
    assert epoch_loss > 0, "Training loss should be positive."

    # 3. Validation / Evaluation
    # Re-use train loader as val loader for demo
    val_dataset = SaltDataset(
        sub_images,
        sub_masks,
        sub_depths,
        sub_ids,
        transform=get_transforms("val"),  # No augmentation
        depth_stats=depth_stats,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=4, shuffle=False, num_workers=0
    )

    print(" - Running evaluation...")
    map_score = evaluate(model, val_loader, device)
    print(f"   Validation mAP: {map_score:.4f}")

    # 4. Marginalized Inference
    print(" - Running marginalized inference...")
    # Use the same loader for inference demo
    pred_ids, pred_rles = predict_marginalized(model, val_loader, device, num_scans=3)

    assert len(pred_ids) == 16, f"Expected 16 predictions, got {len(pred_ids)}"
    assert len(pred_rles) == 16, "Mismatch in RLE count."

    # Verify RLE format
    if len(pred_rles[0]) > 0:
        parts = pred_rles[0].split()
        assert (
            len(parts) % 2 == 0
        ), "RLE string must have even number of elements (start, length pairs)."

    print(" - Pipeline verified successfully.")

    return depth_stats


def generate_demo_submission(depth_stats):
    """Generates a dummy submission file to prove end-to-end capability."""
    print("[5/5] Generating Demo Submission...")

    # Create a dummy submission file in the expected format
    submission_path = "./working/submission_demo.csv"

    # We'll just create a dataframe with the IDs from the test metadata
    test_meta_path = "./metadata/test.csv"
    if os.path.exists(test_meta_path):
        df_test = pd.read_csv(test_meta_path)
        # Take top 5 for demo
        df_test = df_test.head(5)

        # Fake RLEs
        df_test["rle_mask"] = "1 1"

        # Keep only required columns
        df_sub = df_test[["id", "rle_mask"]]
        df_sub.to_csv(submission_path, index=False)
        print(f" - Submission saved to {submission_path}")
        assert os.path.exists(submission_path), "Submission file not created."
    else:
        print(" - Test metadata not found, skipping submission generation.")


if __name__ == "__main__":
    # 1. Setup
    seed_everything(42)

    # 2. Verify Utilities
    test_utils()

    # 3. Verify Model
    model, device = test_model_logic()

    # 4. Verify Loss
    test_loss_function(device)

    # 5. Verify Pipeline (Train/Eval/Inference)
    depth_stats = test_training_and_inference_pipeline(model, device)

    # 6. Generate Submission Artifact
    generate_demo_submission(depth_stats)

    print("\nAll demonstrations completed successfully.")

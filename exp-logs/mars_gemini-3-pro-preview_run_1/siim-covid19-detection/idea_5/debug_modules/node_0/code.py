import os
import torch
import numpy as np
import pandas as pd
import shutil

# Import from the provided library
from library.config import seed_everything, DEVICE, SUBMISSION_PATH, WORKING_DIR
from library.utils import iou_score, get_box_from_mask
from library.dataset import get_dataloaders
from library.model import ResNet18AttentionUNet
from library.engine import train_one_epoch, evaluate, inference


def run_demo():
    # 1. Setup and Reproducibility
    print("--- Step 1: Initialization ---")
    seed_everything(42)
    print(f"Device: {DEVICE}")

    # Clean working directory specific to this run to ensure fresh debug data generation
    # (The library code saves .npy files to WORKING_DIR)
    if os.path.exists(WORKING_DIR):
        # We don't delete the dir itself to avoid permission issues, just files if needed
        # But get_dataloaders(load_cached_data=False) handles overwriting.
        pass

    # 2. Verify Utility Functions
    print("\n--- Step 2: Verifying Utility Functions ---")

    # Test IoU
    box_a = [0, 0, 100, 100]
    box_b = [
        50,
        0,
        150,
        100,
    ]  # Overlap is 50x100 = 5000. Union is 10000 + 10000 - 5000 = 15000. IoU = 1/3
    iou = iou_score(box_a, box_b)
    print(f"IoU Score Test: {iou:.4f}")
    assert abs(iou - 0.3333) < 1e-3, "IoU calculation is incorrect"

    # Test Box from Mask
    dummy_mask = np.zeros((200, 200), dtype=np.float32)
    dummy_mask[50:100, 50:100] = 1.0  # 50x50 square
    boxes = get_box_from_mask(dummy_mask, threshold=0.5)
    print(f"Extracted Boxes: {boxes}")
    assert len(boxes) == 1, "Should extract exactly one box"
    # cv2.boundingRect might return slightly different bounds depending on implementation, but should be close
    b = boxes[0]
    assert b[0] == 50 and b[1] == 50, "Box coordinates incorrect"

    # 3. Data Loading (Debug Mode)
    print("\n--- Step 3: Data Loading (Debug Mode) ---")
    # We use debug=True to load only a small subset (e.g., 20 images)
    # load_cached_data=False ensures we process this subset instead of loading full cached .npy files
    train_loader, val_loader, test_loader = get_dataloaders(
        load_cached_data=False, debug=True, debug_sample_size=16
    )

    batch = next(iter(train_loader))
    images = batch["image"]
    masks = batch["mask"]
    labels = batch["label"]

    print(f"Batch keys: {batch.keys()}")
    print(f"Image Shape: {images.shape}")
    print(f"Mask Shape: {masks.shape}")
    print(f"Label Shape: {labels.shape}")

    # Assertions
    assert images.dim() == 4, "Images should be 4D tensor (B, C, H, W)"
    assert masks.dim() == 4, "Masks should be 4D tensor (B, C, H, W)"
    assert images.shape[1] == 3, "Images should have 3 channels (RGB)"
    assert masks.shape[1] == 1, "Masks should have 1 channel"
    assert (
        labels.shape[0] == images.shape[0]
    ), "Batch size mismatch between images and labels"

    # 4. Model Initialization
    print("\n--- Step 4: Model Initialization ---")
    # pretrained=False to avoid downloading weights during this demo
    model = ResNet18AttentionUNet(num_classes=4, pretrained=False)
    model.to(DEVICE)

    # Forward pass check
    with torch.no_grad():
        dummy_input = torch.randn(2, 3, 512, 512).to(DEVICE)
        cls_out, seg_out = model(dummy_input)

    print(f"Class Output Shape: {cls_out.shape}")
    print(f"Seg Output Shape: {seg_out.shape}")

    assert cls_out.shape == (2, 4), "Classification output shape mismatch"
    assert seg_out.shape == (2, 1, 512, 512), "Segmentation output shape mismatch"

    # 5. Training Loop (One Epoch)
    print("\n--- Step 5: Training (1 Epoch) ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Train for one epoch
    train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, epoch=1)
    print(f"Training Epoch Completed. Loss: {train_loss:.4f}")

    assert isinstance(train_loss, float), "Train loss should be a float"
    assert train_loss > 0, "Train loss should be positive"

    # 6. Evaluation
    print("\n--- Step 6: Evaluation ---")
    val_loss, val_map = evaluate(model, val_loader, DEVICE)
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation mAP: {val_map:.4f}")

    assert isinstance(val_map, float), "mAP should be a float"
    assert 0.0 <= val_map <= 1.0, "mAP should be between 0 and 1"

    # 7. Inference
    print("\n--- Step 7: Inference ---")
    # Run inference on the test loader (debug subset)
    inference(model, test_loader, DEVICE)

    # Verify submission file
    if os.path.exists(SUBMISSION_PATH):
        print(f"Submission file found at {SUBMISSION_PATH}")
        df_sub = pd.read_csv(SUBMISSION_PATH)
        print("Submission Head:")
        print(df_sub.head())

        # Check columns
        assert "id" in df_sub.columns, "Submission missing 'id' column"
        assert (
            "PredictionString" in df_sub.columns
        ), "Submission missing 'PredictionString' column"

        # Check content format
        if len(df_sub) > 0:
            pred_str = df_sub.iloc[0]["PredictionString"]
            assert isinstance(pred_str, str), "PredictionString must be a string"
            # Basic check for format (e.g., 'none ...' or 'opacity ...')
            valid_starts = [
                "none",
                "opacity",
                "negative",
                "typical",
                "indeterminate",
                "atypical",
            ]
            assert any(
                pred_str.startswith(v) for v in valid_starts
            ), f"Invalid prediction string start: {pred_str}"
    else:
        raise FileNotFoundError(f"Submission file was not created at {SUBMISSION_PATH}")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

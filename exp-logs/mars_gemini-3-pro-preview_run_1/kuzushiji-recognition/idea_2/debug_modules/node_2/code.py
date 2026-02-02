import os
import sys
import torch
import numpy as np
import pandas as pd
import cv2
import warnings

# Import from the provided library
from library.config import Config
from library.utils import get_affine_transform, affine_transform, calc_f1_score
from library.dataset import KuzushijiDataset
from library.model import HRNetCenterNet
from library.loss import CenterNetLoss
from library.engine import train_one_epoch, evaluate, load_val_gt_info
from library.inference import generate_submission

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def run_demo():
    print("=== Starting Kuzushiji Library Demo ===\n")

    # 1. Setup and Configuration
    print("--- 1. Configuration Setup ---")
    # Override Config for speed in this demo
    Config.NUM_EPOCHS = 1
    Config.BATCH_SIZE = 2
    Config.NUM_WORKERS = 2
    Config.setup()
    Config.seed_everything(Config.SEED)

    print(f"Device: {Config.DEVICE}")
    print(f"Input Size: {Config.INPUT_SIZE}")
    print(f"Working Dir: {Config.WORKING_DIR}")
    print("Configuration loaded and verified.\n")

    # 2. Verify Utility Functions
    print("--- 2. Verifying Utilities ---")

    # Test Affine Transform
    # Case: Resize 100x100 to 50x50. Center (50, 50) should map to (25, 25).
    src_shape = (100, 100)
    dst_size = 50
    trans = get_affine_transform(src_shape, dst_size)
    pt = [50, 50]
    new_pt = affine_transform(pt, trans)

    # Allow small float error
    assert np.allclose(new_pt, [25, 25], atol=1.0), f"Affine transform failed: {new_pt}"
    print("Affine transform logic verified.")

    # Test F1 Score
    # Perfect match case
    preds = [[{"point": (10, 10), "label": 1, "score": 0.9}]]
    gts = [[{"box": (5, 5, 10, 10), "label": 1}]]  # Box covers (5,5) to (15,15)
    f1 = calc_f1_score(preds, gts)
    assert f1 == 1.0, f"F1 Score calculation failed for perfect match. Got {f1}"

    # No match case (wrong label)
    preds_bad = [[{"point": (10, 10), "label": 2, "score": 0.9}]]
    f1_bad = calc_f1_score(preds_bad, gts)
    assert f1_bad == 0.0, f"F1 Score calculation failed for mismatch. Got {f1_bad}"
    print("F1 Score metric verified.\n")

    # 3. Verify Dataset
    print("--- 3. Verifying Dataset ---")
    debug_size = 10
    ds = KuzushijiDataset(split="train", debug_size=debug_size)
    print(f"Dataset loaded with {len(ds)} samples (debug mode).")

    # Fetch one item
    item = ds[0]
    img = item["image"]
    hm = item["hm"]
    wh = item["wh"]
    reg = item["reg"]

    # Check shapes
    # Image: (3, 1024, 1024)
    assert img.shape == (
        3,
        Config.INPUT_SIZE,
        Config.INPUT_SIZE,
    ), f"Incorrect image shape: {img.shape}"
    # Heatmap: (Num_Classes, 256, 256)
    assert hm.shape == (
        Config.NUM_CLASSES,
        Config.INPUT_SIZE // 4,
        Config.INPUT_SIZE // 4,
    ), f"Incorrect heatmap shape: {hm.shape}"
    # WH/Reg: (Max_Preds, 2)
    assert wh.shape == (Config.MAX_PREDS, 2), f"Incorrect WH shape: {wh.shape}"

    print("Dataset shapes verified.\n")

    # 4. Verify Model and Loss
    print("--- 4. Verifying Model & Loss ---")
    model = HRNetCenterNet().to(Config.DEVICE)
    criterion = CenterNetLoss()

    # Create a dummy batch from the dataset item
    # Add batch dimension
    batch = {
        "image": item["image"].unsqueeze(0).to(Config.DEVICE),
        "hm": item["hm"].unsqueeze(0).to(Config.DEVICE),
        "wh": item["wh"].unsqueeze(0).to(Config.DEVICE),
        "reg": item["reg"].unsqueeze(0).to(Config.DEVICE),
        "ind": item["ind"].unsqueeze(0).to(Config.DEVICE),
        "reg_mask": item["reg_mask"].unsqueeze(0).to(Config.DEVICE),
        "image_id": [item["image_id"]],
    }

    # Forward pass
    outputs = model(batch["image"])
    hm_pred, wh_pred, reg_pred = outputs

    # Verify output shapes
    # hm_pred: (B, Num_Classes, H/4, W/4)
    expected_hm_shape = (
        1,
        Config.NUM_CLASSES,
        Config.INPUT_SIZE // 4,
        Config.INPUT_SIZE // 4,
    )
    assert (
        hm_pred.shape == expected_hm_shape
    ), f"Model HM output shape mismatch: {hm_pred.shape}"

    # Calculate Loss
    loss, l_hm, l_wh, l_reg = criterion(outputs, batch)

    assert not torch.isnan(loss), "Loss returned NaN"
    assert loss.item() > 0, "Loss should be positive"

    print(f"Forward pass successful. Total Loss: {loss.item():.4f}")
    print(
        f"Components - HM: {l_hm.item():.4f}, WH: {l_wh.item():.4f}, Reg: {l_reg.item():.4f}\n"
    )

    # 5. Verify Training Engine (One Epoch)
    print("--- 5. Running Training Loop (1 Epoch) ---")

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(
        ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing overhead in demo
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)

    # Run training step
    avg_loss = train_one_epoch(
        model, optimizer, train_loader, Config.DEVICE, criterion, epoch=0
    )
    print(f"Training epoch completed. Average Loss: {avg_loss:.4f}")

    # Run validation step
    # Need validation metadata cache first
    val_gt_data = load_val_gt_info(Config.VAL_METADATA_PATH)
    val_ds = KuzushijiDataset(split="val", debug_size=debug_size)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    val_loss, val_f1 = evaluate(
        model, val_loader, Config.DEVICE, criterion, val_gt_data
    )
    print(f"Validation completed. Loss: {val_loss:.4f}, F1: {val_f1:.4f}\n")

    # Save model manually for inference step (since F1 might be 0.0 and not trigger save in a full loop)
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} for inference testing.\n")

    # 6. Verify Inference
    print("--- 6. Running Inference ---")

    # Run generation
    # We use debug=True to only run on a subset of test images
    generate_submission(model, Config.DEVICE, debug=True)

    # Check if submission file exists
    if os.path.exists(Config.SUBMISSION_PATH):
        df = pd.read_csv(Config.SUBMISSION_PATH)
        print(f"Submission file generated at {Config.SUBMISSION_PATH}")
        print(f"Rows in submission: {len(df)}")
        print("Sample row:")
        print(df.head(1))

        # Validate format
        assert (
            "image_id" in df.columns and "labels" in df.columns
        ), "Submission columns missing"
    else:
        raise FileNotFoundError("Submission file was not created.")

    print("\n=== Demo Completed Successfully ===")


if __name__ == "__main__":
    run_demo()

import os
import torch
import pandas as pd
import numpy as np
import cv2
import torch.optim as optim
from torch.utils.data import DataLoader

# Import library modules
from library.model import ResNet34WideLinkNet
from library.dataset import SaltDataset, get_depth_stats, get_transforms
from library.engine import (
    train_model,
    evaluate,
    predict_with_tta,
    generate_submission,
    set_seed,
    center_crop,
)
from library.utils import calculate_iou_batch
from library.losses import CombinedLoss


def main():
    # 1. Configuration & Setup
    SEED = 42
    set_seed(SEED)

    INPUT_DIR = "./input"
    METADATA_DIR = "./metadata"
    WORKING_DIR = "./working"
    PSEUDO_MASK_DIR = os.path.join(WORKING_DIR, "pseudo_masks")
    os.makedirs(PSEUDO_MASK_DIR, exist_ok=True)
    SUBMISSION_DIR = "./submission"
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 32
    NUM_WORKERS = 2
    STAGE1_EPOCHS = 35
    STAGE2_EPOCHS = 35
    PATIENCE = 8
    LR = 1e-4

    print(f"Using device: {DEVICE}")

    # 2. Data Preparation (Stage 1)
    print("\n--- Stage 1: Data Preparation ---")
    depth_mean, depth_std = get_depth_stats(os.path.join(METADATA_DIR, "train.csv"))
    print(f"Depth Stats - Mean: {depth_mean:.4f}, Std: {depth_std:.4f}")

    # Train Dataset
    train_dataset = SaltDataset(
        mode="train",
        metadata_file=os.path.join(METADATA_DIR, "train.csv"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir=INPUT_DIR,
    )

    # Val Dataset
    val_dataset = SaltDataset(
        mode="val",
        metadata_file=os.path.join(METADATA_DIR, "val.csv"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir=INPUT_DIR,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # 3. Stage 1 Training
    print("\n--- Stage 1: Robust Supervised Training ---")
    model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    stage1_thresh = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        DEVICE,
        num_epochs=STAGE1_EPOCHS,
        patience=PATIENCE,
        output_dir=WORKING_DIR,
    )

    # Rename best model to stage1_model.pth
    os.rename(
        os.path.join(WORKING_DIR, "best_model.pth"),
        os.path.join(WORKING_DIR, "stage1_model.pth"),
    )
    print(f"Stage 1 Complete. Best Threshold: {stage1_thresh}")

    # 4. Pseudo-Labeling
    print("\n--- Generating Pseudo-Labels ---")
    # Load Stage 1 Model
    model.load_state_dict(torch.load(os.path.join(WORKING_DIR, "stage1_model.pth")))

    # Test Dataset
    test_dataset = SaltDataset(
        mode="test",
        metadata_file=os.path.join(METADATA_DIR, "test.csv"),
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir=INPUT_DIR,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    # Predict (Enforces depth=0 for Generalist Mode)
    test_preds, test_ids = predict_with_tta(model, test_loader, DEVICE)

    # Prepare Metadata for Combined Training
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))

    # Convert train paths to absolute to handle mixed directories
    train_df["image_path"] = train_df["image_path"].apply(
        lambda x: os.path.abspath(os.path.join(INPUT_DIR, x))
    )
    train_df["mask_path"] = train_df["mask_path"].apply(
        lambda x: os.path.abspath(os.path.join(INPUT_DIR, x))
    )

    pseudo_rows = []

    for i, img_id in enumerate(test_ids):
        # Threshold prediction to create binary mask
        mask = (test_preds[i] > stage1_thresh).astype(np.uint8) * 255

        # Save mask to working directory
        mask_filename = f"{img_id}.png"
        mask_path = os.path.join(PSEUDO_MASK_DIR, mask_filename)
        cv2.imwrite(mask_path, mask)

        # Get info from test_df
        row = test_df[test_df["id"] == img_id].iloc[0]

        pseudo_rows.append(
            {
                "id": img_id,
                "rle_mask": "",  # Not used by dataset loader
                "z": row["z"],
                "image_path": os.path.abspath(
                    os.path.join(INPUT_DIR, row["image_path"])
                ),
                "mask_path": os.path.abspath(mask_path),
                "salt_pixels": 0,  # Dummy
                "salt_coverage": 0.0,  # Dummy
                "coverage_class": 0,  # Dummy
            }
        )

    pseudo_df = pd.DataFrame(pseudo_rows)
    combined_df = pd.concat([train_df, pseudo_df], ignore_index=True)

    pseudo_csv_path = os.path.join(WORKING_DIR, "pseudo_train.csv")
    combined_df.to_csv(pseudo_csv_path, index=False)
    print(f"Pseudo-labeled dataset created with {len(combined_df)} samples.")

    # 5. Stage 2 Training
    print("\n--- Stage 2: Retraining with Pseudo-Labels ---")

    # Combined Dataset
    # root_dir="" because we used absolute paths in the CSV
    # Explicitly pass transform to ensure augmentations are applied
    train_transform = get_transforms("train")

    combined_dataset = SaltDataset(
        mode="train_stage2",  # Unique mode to avoid cache collision
        metadata_file=pseudo_csv_path,
        depth_mean=depth_mean,
        depth_std=depth_std,
        root_dir="",
        transform=train_transform,
    )

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        drop_last=True,
    )

    # Re-initialize model from scratch
    model = ResNet34WideLinkNet().to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

    stage2_thresh = train_model(
        model,
        combined_loader,
        val_loader,  # Validate on original validation set
        optimizer,
        DEVICE,
        num_epochs=STAGE2_EPOCHS,
        patience=PATIENCE,
        output_dir=WORKING_DIR,
    )

    # 6. Final Evaluation
    print("\n--- Final Evaluation ---")
    # Load best model from Stage 2
    model.load_state_dict(torch.load(os.path.join(WORKING_DIR, "best_model.pth")))

    loss_fn = CombinedLoss()
    val_loss, val_map, val_thresh = evaluate(model, val_loader, DEVICE, loss_fn)

    print(f"Final Validation Metric: {val_map}")

    # 7. Failure Analysis
    print("\n--- Failure Analysis ---")
    model.eval()
    ious = []
    depths_val = []

    with torch.no_grad():
        for batch in val_loader:
            images = batch[0].to(DEVICE)
            masks = batch[1].to(DEVICE)
            depth_vals = batch[2].cpu().numpy()  # standardized

            # Predict
            logits = model(images, batch[2].to(DEVICE))
            probs = torch.sigmoid(logits)

            # Crop to original size for metric calculation
            probs_c = center_crop(probs, 101, 101)
            masks_c = center_crop(masks, 101, 101)

            # Binarize with best threshold
            preds_bin = (probs_c > val_thresh).float()

            # Calculate IoU per image
            batch_iou = calculate_iou_batch(
                masks_c.cpu().numpy(), preds_bin.cpu().numpy()
            )

            ious.extend(batch_iou)

            # De-standardize depth for interpretation
            raw_depths = (depth_vals * depth_std) + depth_mean
            depths_val.extend(raw_depths)

    ious = np.array(ious)
    depths_val = np.array(depths_val)
    errors = 1.0 - ious

    # Calculate correlation
    if np.std(errors) > 0 and np.std(depths_val) > 0:
        corr = np.corrcoef(errors, depths_val)[0, 1]
    else:
        corr = 0.0
    print(f"Correlation between Error (1-IoU) and Depth: {corr:.4f}")

    # 8. Submission
    if val_map > 0.7985:
        print("\n--- Generating Submission ---")
        preds, ids = predict_with_tta(model, test_loader, DEVICE)

        final_sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
        generate_submission(preds, ids, val_thresh, final_sub_path)
    else:
        print(
            f"Validation metric {val_map} did not meet threshold 0.7985. Skipping submission."
        )


if __name__ == "__main__":
    main()

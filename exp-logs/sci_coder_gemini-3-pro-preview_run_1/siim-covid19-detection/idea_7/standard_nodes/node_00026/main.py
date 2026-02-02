import os
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pydicom

# Import from provided library
from library.config import Config, seed_everything
from library.dataset import get_loaders
from library.model import ResNet18UNet
from library.engine import train_one_epoch, valid_one_epoch
from library.utils import get_bbox_from_mask


def run():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Create submission directory
    os.makedirs("./submission", exist_ok=True)

    print(f"Device: {device}")

    # 2. Data Loading
    # Using load_cached_data=True to speed up if available
    train_loader, val_loader, test_loader = get_loaders(
        load_cached_data=True,
        debug=Config.DEBUG,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model
    model = ResNet18UNet(pretrained=Config.PRETRAINED)
    model = model.to(device)

    # 4. Optimizer & Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
    )

    # 5. Training Loop
    best_composite_score = 0.0

    print("Starting training...")
    for epoch in range(1, Config.EPOCHS + 1):
        # Train
        train_loss = train_one_epoch(
            model, optimizer, scheduler, train_loader, device, epoch
        )

        # Validation
        val_metrics = valid_one_epoch(model, val_loader, device, epoch)

        # Scheduler Step
        scheduler.step()

        # Checkpoint
        if val_metrics["composite_score"] > best_composite_score:
            best_composite_score = val_metrics["composite_score"]
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"New best model saved! Score: {best_composite_score:.4f}")

    print(f"Training complete. Best Composite Score: {best_composite_score:.4f}")

    # 6. Final Validation & Failure Analysis
    print("\nRunning Final Validation & Failure Analysis...")
    model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
    model.eval()

    # Load validation metadata for analysis
    val_df = pd.read_csv(Config.VAL_CSV)
    if Config.DEBUG:
        val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    criterion_cls = nn.CrossEntropyLoss(reduction="none")

    all_losses = []

    # Re-run validation inference to get per-sample errors for analysis
    with torch.no_grad():
        for i, (images, masks, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward
            logit_cls, _ = model(images)

            # Calculate classification loss per sample
            target_cls = torch.argmax(labels, dim=1)
            loss_cls = criterion_cls(logit_cls, target_cls)

            all_losses.extend(loss_cls.cpu().numpy())

    # Add losses to dataframe
    min_len = min(len(all_losses), len(val_df))
    val_df = val_df.iloc[:min_len].copy()
    val_df["error_magnitude"] = all_losses[:min_len]

    # Calculate final mAP on full validation set
    final_metrics = valid_one_epoch(model, val_loader, device, epoch="FINAL")
    final_map = final_metrics["map_score"]

    # Required Output Format
    print(f"Final Validation Metric: {final_map}")

    # Failure Analysis: Correlation
    label_cols = [
        "Negative for Pneumonia",
        "Typical Appearance",
        "Indeterminate Appearance",
        "Atypical Appearance",
    ]

    print("\nFailure Analysis (Correlation with Error Magnitude):")
    for col in label_cols:
        if col in val_df.columns:
            corr = val_df["error_magnitude"].corr(val_df[col])
            print(f"  {col}: {corr:.4f}")

    # 7. Submission
    THRESHOLD = 0.49944536565378

    if final_map > THRESHOLD:
        print(
            f"\nMetric ({final_map}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        test_df = pd.read_csv(Config.TEST_CSV)
        submission_rows = []

        # Iterate through test loader
        with torch.no_grad():
            for batch_idx, (images, _, _) in enumerate(test_loader):
                images = images.to(device)

                # Original Forward
                logit_cls, logit_seg = model(images)
                prob_cls = torch.softmax(logit_cls, dim=1)
                prob_seg = torch.sigmoid(logit_seg)

                # TTA: Horizontal Flip
                images_flipped = torch.flip(images, dims=[3])
                logit_cls_tta, logit_seg_tta = model(images_flipped)
                prob_cls_tta = torch.softmax(logit_cls_tta, dim=1)
                prob_seg_tta = torch.sigmoid(logit_seg_tta)

                # Flip mask back
                prob_seg_tta = torch.flip(prob_seg_tta, dims=[3])

                # Average
                avg_prob_cls = (prob_cls + prob_cls_tta) / 2.0
                avg_prob_seg = (prob_seg + prob_seg_tta) / 2.0

                # Process batch
                bs = images.size(0)
                for i in range(bs):
                    # Get corresponding metadata row
                    global_idx = batch_idx * Config.BATCH_SIZE + i
                    if global_idx >= len(test_df):
                        break

                    row = test_df.iloc[global_idx]
                    study_id = row["study_id"]
                    image_id = row["image_id"]

                    # Study Prediction
                    cls_idx = torch.argmax(avg_prob_cls[i]).item()
                    cls_conf = avg_prob_cls[i, cls_idx].item()
                    cls_label = Config.STUDY_LABELS[cls_idx]

                    # Study Label Formatting
                    label_map = {
                        "Negative for Pneumonia": "negative",
                        "Typical Appearance": "typical",
                        "Indeterminate Appearance": "indeterminate",
                        "Atypical Appearance": "atypical",
                    }
                    pred_label_str = label_map[cls_label]

                    study_pred_string = f"{pred_label_str} {cls_conf:.4f} 0 0 1 1"

                    # Image Prediction & Gating Logic
                    if pred_label_str == "negative":
                        image_pred_string = "none 1 0 0 1 1"
                    else:
                        # Extract boxes
                        mask_np = avg_prob_seg[i, 0].cpu().numpy()
                        binary_mask = (mask_np > 0.5).astype(np.uint8)
                        boxes = get_bbox_from_mask(binary_mask)

                        # Get original dimensions for scaling
                        dcm_path = os.path.join(Config.INPUT_DIR, row["file_path"])
                        try:
                            dcm = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                            orig_h, orig_w = dcm.Rows, dcm.Columns
                        except:
                            orig_h, orig_w = Config.IMG_SIZE, Config.IMG_SIZE

                        scale_x = orig_w / Config.IMG_SIZE
                        scale_y = orig_h / Config.IMG_SIZE

                        scaled_boxes = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            # Compute box confidence (mean prob inside box)
                            box_conf = np.mean(mask_np[y1:y2, x1:x2])

                            sx1 = x1 * scale_x
                            sy1 = y1 * scale_y
                            sx2 = x2 * scale_x
                            sy2 = y2 * scale_y
                            scaled_boxes.append([sx1, sy1, sx2, sy2, box_conf])

                        # Format string
                        if not scaled_boxes:
                            image_pred_string = "none 1 0 0 1 1"
                        else:
                            parts = []
                            for b in scaled_boxes:
                                parts.append(
                                    f"opacity {b[4]:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                                )
                            image_pred_string = " ".join(parts)

                    # Append rows
                    submission_rows.append(
                        {
                            "id": f"{study_id}_study",
                            "PredictionString": study_pred_string,
                        }
                    )
                    submission_rows.append(
                        {
                            "id": f"{image_id}_image",
                            "PredictionString": image_pred_string,
                        }
                    )

        # Save submission
        sub_df = pd.DataFrame(submission_rows)
        sub_path = "./submission/submission.csv"
        sub_df.to_csv(sub_path, index=False)
        print(f"Submission saved to {sub_path}")

    else:
        print(f"Metric ({final_map}) <= Threshold ({THRESHOLD}). Skipping submission.")


if __name__ == "__main__":
    run()

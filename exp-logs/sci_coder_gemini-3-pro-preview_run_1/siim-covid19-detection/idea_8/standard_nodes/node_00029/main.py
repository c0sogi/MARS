import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import cv2
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.stats import pearsonr

from library.config import Config
from library.utils import (
    seed_everything,
    mask2bbox,
    get_map_score,
    post_process_submission,
)
from library.data import get_dataloaders
from library.model import ResNet18UNetASPP
from library.engine import fit


def calculate_iou(pred_mask, true_mask):
    """Calculates IoU for a single sample."""
    intersection = (pred_mask * true_mask).sum()
    union = pred_mask.sum() + true_mask.sum() - intersection
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = Config.DEVICE
    print(f"Running on device: {device}")

    # 2. Data Loading
    # Using full dataset to ensure metric threshold is met.
    # The dataset is small (~5k images), so 20 epochs is fast on A100.
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
        load_cached_data=True,
    )

    # 3. Model Initialization
    model = ResNet18UNetASPP(num_classes=Config.NUM_CLASSES).to(device)

    # 4. Optimizer & Scheduler
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.SCHEDULER_MIN_LR
    )

    # 5. Training
    print("Starting training...")
    fit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=Config.EPOCHS,
    )

    # 6. Final Validation & Failure Analysis
    print("\n==== Final Validation & Failure Analysis ====")
    # Load best model
    if os.path.exists(Config.BEST_MODEL_PATH):
        model.load_state_dict(torch.load(Config.BEST_MODEL_PATH, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: Best model not found, using current weights.")

    model.eval()

    # Containers for metric calculation
    pred_boxes_list = []
    pred_scores_list = []
    true_boxes_list = []

    # Containers for failure analysis
    error_magnitudes = []  # 1 - IoU
    target_areas = []
    num_boxes_list = []

    # Load metadata for feature extraction
    df_val = pd.read_csv(Config.VAL_CSV)
    # Create a map for fast lookup: image_id -> row
    # Note: df_val has 'image_id' which matches the loader
    meta_map = df_val.set_index("image_id")

    with torch.no_grad():
        for batch in val_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)
            image_ids = batch["image_id"]

            # Inference
            _, logits_seg = model(images)
            probs_seg = torch.sigmoid(logits_seg)

            # Process batch
            probs_seg_np = probs_seg.cpu().numpy()
            masks_np = masks.cpu().numpy()

            for i in range(len(images)):
                # Metric Data
                p_mask = probs_seg_np[i][0]
                t_mask = masks_np[i][0]

                p_boxes = mask2bbox(p_mask, threshold=0.5)

                # Calculate scores for boxes
                p_scores = []
                for box in p_boxes:
                    x1, y1, x2, y2 = box
                    # Clamp coords
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(p_mask.shape[1], x2), min(p_mask.shape[0], y2)
                    if x2 > x1 and y2 > y1:
                        p_scores.append(np.mean(p_mask[y1:y2, x1:x2]))
                    else:
                        p_scores.append(0.0)

                t_boxes = mask2bbox(t_mask, threshold=0.5)

                pred_boxes_list.append(p_boxes)
                pred_scores_list.append(p_scores)
                true_boxes_list.append(t_boxes)

                # Failure Analysis Data
                # Error: 1 - IoU (using 0.5 threshold for binary mask)
                bin_p_mask = (p_mask > 0.5).astype(np.float32)
                iou = calculate_iou(bin_p_mask, t_mask)
                error_magnitudes.append(1.0 - iou)

                # Features from metadata
                img_id = image_ids[i]
                if img_id in meta_map.index:
                    # Handle potential duplicate indices if any, though metadata generation ensures uniqueness usually
                    row = meta_map.loc[img_id]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    # Target Area (sum of mask pixels)
                    target_areas.append(t_mask.sum())
                    # Num Boxes
                    num_boxes_list.append(len(t_boxes))
                else:
                    target_areas.append(0)
                    num_boxes_list.append(0)

    # Calculate Metric
    final_map = get_map_score(pred_boxes_list, pred_scores_list, true_boxes_list)
    print(f"Final Validation Metric: {final_map}")

    # Calculate Correlations
    if len(error_magnitudes) > 1:
        corr_area, _ = pearsonr(error_magnitudes, target_areas)
        corr_boxes, _ = pearsonr(error_magnitudes, num_boxes_list)
        print(f"Correlation (Error vs Target Area): {corr_area:.4f}")
        print(f"Correlation (Error vs Num Boxes): {corr_boxes:.4f}")

    # 7. Submission
    THRESHOLD = 0.49944536565378

    if final_map > THRESHOLD:
        print(
            f"\nMetric ({final_map}) > Threshold ({THRESHOLD}). Generating submission..."
        )

        study_preds_all = []
        image_preds_all = []
        study_ids_all = []
        image_ids_all = []

        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(device)
                study_ids = batch["study_id"]
                image_ids = batch["image_id"]

                # TTA: Original
                logits_cls_1, logits_seg_1 = model(images)

                # TTA: Horizontal Flip
                images_flipped = torch.flip(images, [3])
                logits_cls_2, logits_seg_2 = model(images_flipped)

                # Flip segmentation back
                logits_seg_2 = torch.flip(logits_seg_2, [3])

                # Average
                probs_cls = (
                    torch.softmax(logits_cls_1, dim=1)
                    + torch.softmax(logits_cls_2, dim=1)
                ) / 2.0
                probs_seg = (
                    torch.sigmoid(logits_seg_1) + torch.sigmoid(logits_seg_2)
                ) / 2.0

                probs_cls_np = probs_cls.cpu().numpy()
                probs_seg_np = probs_seg.cpu().numpy()

                for i in range(len(images)):
                    # Study Prediction
                    s_probs = probs_cls_np[i]
                    study_preds_all.append(s_probs)
                    study_ids_all.append(study_ids[i])

                    # Image Prediction (Gated)
                    # Class 0 is "Negative for Pneumonia"
                    pred_label_idx = np.argmax(s_probs)

                    if pred_label_idx == 0:
                        # Gating: Force no boxes
                        image_preds_all.append({"boxes": [], "scores": []})
                    else:
                        # Extract boxes
                        p_mask = probs_seg_np[i][0]
                        boxes = mask2bbox(p_mask, threshold=0.5)

                        scores = []
                        for box in boxes:
                            x1, y1, x2, y2 = box
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(p_mask.shape[1], x2), min(p_mask.shape[0], y2)
                            if x2 > x1 and y2 > y1:
                                scores.append(float(np.mean(p_mask[y1:y2, x1:x2])))
                            else:
                                scores.append(0.0)

                        image_preds_all.append({"boxes": boxes, "scores": scores})

                    image_ids_all.append(image_ids[i])

        # Save Submission
        post_process_submission(
            study_preds_all,
            study_ids_all,
            image_preds_all,
            image_ids_all,
            save_path=Config.SUBMISSION_FILE,
        )
        print(f"Submission saved to {Config.SUBMISSION_FILE}")

    else:
        print(
            f"\nMetric ({final_map}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.ops as ops

# Import from the provided library
from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.engine import train_model, set_seed
from library.utils import collate_fn
from library.submission import generate_submission


def calculate_map(model, dataloader, device):
    """
    Calculates the PASCAL VOC 2010 mean Average Precision (mAP) at IoU > 0.5
    for the 'opacity' class.
    """
    model.eval()

    # Containers for all predictions and ground truths
    # Format: dict mapping image_index to list of boxes
    gt_boxes_map = {}
    gt_matched_map = {}  # To track which GT boxes have been matched

    # List of all predictions: (score, image_index, box_tensor)
    all_predictions = []

    image_idx_offset = 0

    with torch.no_grad():
        for images, targets, _ in dataloader:
            images = list(img.to(device) for img in images)

            # Forward pass
            detections, _ = model(images)

            for i, (target, detection) in enumerate(zip(targets, detections)):
                global_idx = image_idx_offset + i

                # Process Ground Truth
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                # Filter for 'opacity' class (class ID 1)
                opacity_indices = gt_labels == 1
                gt_opacity_boxes = gt_boxes[opacity_indices]

                gt_boxes_map[global_idx] = gt_opacity_boxes
                gt_matched_map[global_idx] = np.zeros(len(gt_opacity_boxes), dtype=bool)

                # Process Predictions
                pred_boxes = detection["boxes"].cpu().numpy()
                pred_scores = detection["scores"].cpu().numpy()
                pred_labels = detection["labels"].cpu().numpy()

                # Filter for 'opacity' class (class ID 1)
                pred_indices = pred_labels == 1
                valid_boxes = pred_boxes[pred_indices]
                valid_scores = pred_scores[pred_indices]

                for box, score in zip(valid_boxes, valid_scores):
                    all_predictions.append((score, global_idx, box))

            image_idx_offset += len(images)

    # Sort all predictions by score descending
    all_predictions.sort(key=lambda x: x[0], reverse=True)

    true_positives = np.zeros(len(all_predictions))
    false_positives = np.zeros(len(all_predictions))

    for i, (score, img_idx, pred_box) in enumerate(all_predictions):
        gt_boxes = gt_boxes_map[img_idx]
        matched = gt_matched_map[img_idx]

        if len(gt_boxes) == 0:
            false_positives[i] = 1
            continue

        # Calculate IoU with all GT boxes in this image
        # pred_box: [x1, y1, x2, y2]
        # gt_boxes: [N, 4]

        # Manually compute IoU or use torchvision
        pb_tensor = torch.tensor(pred_box).unsqueeze(0)
        gt_tensor = torch.tensor(gt_boxes)
        ious = ops.box_iou(pb_tensor, gt_tensor).squeeze(0).numpy()

        max_iou = -1
        max_idx = -1

        if len(ious) > 0:
            max_iou = np.max(ious)
            max_idx = np.argmax(ious)

        if max_iou > 0.5:
            if not matched[max_idx]:
                true_positives[i] = 1
                matched[max_idx] = True
            else:
                false_positives[i] = 1  # Duplicate detection
        else:
            false_positives[i] = 1

    # Compute Precision and Recall
    cum_tp = np.cumsum(true_positives)
    cum_fp = np.cumsum(false_positives)

    total_positives = sum([len(b) for b in gt_boxes_map.values()])

    if total_positives == 0:
        return 0.0

    recall = cum_tp / total_positives
    precision = cum_tp / (cum_tp + cum_fp + 1e-6)

    # Compute AP (PASCAL VOC 2010 uses all points interpolation)
    # Append sentinel values
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def perform_failure_analysis(model, dataloader, dataset_df, device):
    """
    Analyzes correlation between study classification error and image metadata.
    """
    print("\n--- Failure Analysis ---")
    model.eval()

    errors = []
    widths = []
    heights = []
    aspect_ratios = []

    # Create a map from image_id (str) to metadata rows
    # dataset_df has 'image_id' column
    meta_map = dataset_df.set_index("image_id")

    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = list(img.to(device) for img in images)

            # Forward pass: get study probs
            _, study_probs = model(images)
            study_preds = torch.argmax(study_probs, dim=1).cpu().numpy()

            # Get ground truth study labels
            # targets is list of dicts
            gt_labels = torch.stack([t["study_label"] for t in targets]).cpu().numpy()

            # Calculate errors
            batch_errors = (study_preds != gt_labels).astype(int)

            for i, img_id in enumerate(image_ids):
                # img_id is the string ID from dataset
                if img_id in meta_map.index:
                    row = meta_map.loc[img_id]
                    # Handle case where index might return multiple rows (shouldn't happen with unique image_id)
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]

                    w = row["width"]
                    h = row["height"]

                    errors.append(batch_errors[i])
                    widths.append(w)
                    heights.append(h)
                    aspect_ratios.append(w / h)

    # Calculate correlations
    if len(errors) > 0:
        corr_width = np.corrcoef(errors, widths)[0, 1]
        corr_height = np.corrcoef(errors, heights)[0, 1]
        corr_ar = np.corrcoef(errors, aspect_ratios)[0, 1]

        print(f"Correlation (Error vs Width): {corr_width:.4f}")
        print(f"Correlation (Error vs Height): {corr_height:.4f}")
        print(f"Correlation (Error vs Aspect Ratio): {corr_ar:.4f}")
    else:
        print("Insufficient data for failure analysis.")


def main():
    # 1. Configuration and Setup
    set_seed(Config.SEED)
    device = Config.DEVICE

    # Use Config values (or slightly reduced for the 24h limit if needed, but 12 epochs should fit)
    Config.NUM_EPOCHS = 12
    Config.BATCH_SIZE = 16

    print(f"Running on device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset = SIIMDataset(split="train", load_cached_data=True)
    val_dataset = SIIMDataset(split="val", load_cached_data=True)

    # REMOVED SUBSETTING to use full dataset
    print(f"Training on full dataset: {len(train_dataset)} samples.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = MultiTaskFasterRCNN()
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Added Scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=2, verbose=True
    )

    # 4. Training
    print("Starting training...")
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=Config.NUM_EPOCHS,
        patience=Config.PATIENCE,
        save_path=Config.MODEL_SAVE_PATH,
    )

    # 5. Validation Metric
    print("Calculating final validation metric...")
    # Load best model for evaluation
    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    map_score = calculate_map(model, val_loader, device)
    print(f"Final Validation Metric: {map_score}")

    # 6. Failure Analysis
    perform_failure_analysis(model, val_loader, val_dataset.df, device)

    # 7. Submission
    print("Generating submission...")
    generate_submission(load_cached_data=True)


if __name__ == "__main__":
    main()

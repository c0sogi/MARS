import os
import sys
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F

# Import from provided library files
from library.config import Config
from library.utils import set_seed
from library.dataset import get_datasets, get_test_dataset
from library.model import get_model
from library.engine import train_one_epoch, evaluate, generate_submission

# ==========================================
# Configuration Override for Fast Baseline
# ==========================================
# Reduce epochs to ensure execution within time limits while maintaining sufficient convergence.
Config.NUM_EPOCHS = 4


def calculate_map(model, data_loader, device):
    """
    Calculates the Mean Average Precision (mAP) at IoU > 0.5 for the 'opacity' class.
    Implements the standard PASCAL VOC metric logic (Area under Precision-Recall Curve).
    """
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = list(img.to(device) for img in images)

            # Forward pass: returns (detections, study_probs)
            detections, _ = model(images)

            for i in range(len(images)):
                # 1. Process Predictions
                # Move to CPU and numpy
                pred_boxes = detections[i]["boxes"].cpu().numpy()
                pred_scores = detections[i]["scores"].cpu().numpy()

                # Filter by confidence to reduce noise for metric calculation
                # (Though mAP usually handles low conf via sorting, removing very low conf speeds it up)
                mask = pred_scores > 0.05
                pred_boxes = pred_boxes[mask]
                pred_scores = pred_scores[mask]

                all_preds.append({"boxes": pred_boxes, "scores": pred_scores})

                # 2. Process Targets
                gt_boxes = targets[i]["boxes"].numpy()
                # We only care about the boxes (class is always 'opacity' for this metric)
                all_targets.append({"boxes": gt_boxes})

    # Compute AP for the single 'opacity' class
    overall_preds = []  # List of (score, is_tp_flag)
    total_gt_boxes = 0

    for i in range(len(all_preds)):
        p_boxes = all_preds[i]["boxes"]
        p_scores = all_preds[i]["scores"]
        t_boxes = all_targets[i]["boxes"]

        total_gt_boxes += len(t_boxes)

        if len(p_boxes) == 0:
            continue

        if len(t_boxes) == 0:
            # All predictions are False Positives
            for score in p_scores:
                overall_preds.append((score, 0))
            continue

        # Sort predictions by score (descending)
        sorted_idxs = np.argsort(-p_scores)
        p_boxes = p_boxes[sorted_idxs]
        p_scores = p_scores[sorted_idxs]

        # Track which GT boxes have been matched
        gt_matched = np.zeros(len(t_boxes), dtype=bool)

        for j, p_box in enumerate(p_boxes):
            # Calculate IoU between this pred and all GT boxes
            # Intersection
            ixmin = np.maximum(p_box[0], t_boxes[:, 0])
            iymin = np.maximum(p_box[1], t_boxes[:, 1])
            ixmax = np.minimum(p_box[2], t_boxes[:, 2])
            iymax = np.minimum(p_box[3], t_boxes[:, 3])

            iw = np.maximum(ixmax - ixmin, 0.0)
            ih = np.maximum(iymax - iymin, 0.0)
            inters = iw * ih

            # Union
            p_area = (p_box[2] - p_box[0]) * (p_box[3] - p_box[1])
            t_area = (t_boxes[:, 2] - t_boxes[:, 0]) * (t_boxes[:, 3] - t_boxes[:, 1])
            uni = p_area + t_area - inters

            ious = inters / uni

            if len(ious) > 0:
                max_iou_idx = np.argmax(ious)
                max_iou = ious[max_iou_idx]

                if max_iou > 0.5:
                    if not gt_matched[max_iou_idx]:
                        # True Positive
                        overall_preds.append((p_scores[j], 1))
                        gt_matched[max_iou_idx] = True
                    else:
                        # Duplicate detection (False Positive)
                        overall_preds.append((p_scores[j], 0))
                else:
                    # Localization error (False Positive)
                    overall_preds.append((p_scores[j], 0))
            else:
                overall_preds.append((p_scores[j], 0))

    if total_gt_boxes == 0:
        return 0.0

    # Sort all predictions by score to compute PR curve
    overall_preds.sort(key=lambda x: x[0], reverse=True)

    # Extract TP flags
    tps = np.array([x[1] for x in overall_preds])
    fps = 1 - tps

    tp_cumsum = np.cumsum(tps)
    fp_cumsum = np.cumsum(fps)

    recalls = tp_cumsum / total_gt_boxes
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Add sentinel values for integration (0,0) and (1,0)
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute Precision Envelope (make monotonic)
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = np.maximum(precisions[i - 1], precisions[i])

    # Integrate Area Under Curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

    return ap


def failure_analysis(model, data_loader, device):
    """
    Analyzes model failures by correlating localization error (1 - IoU) with object size.
    """
    model.eval()
    errors = []
    areas = []

    with torch.no_grad():
        for images, targets, _ in data_loader:
            images = list(img.to(device) for img in images)
            detections, _ = model(images)

            for i in range(len(images)):
                gt_boxes = targets[i]["boxes"].numpy()
                if len(gt_boxes) == 0:
                    continue

                pred_boxes = detections[i]["boxes"].cpu().numpy()

                # For each GT box, find the best matching prediction
                for box in gt_boxes:
                    box_area = (box[2] - box[0]) * (box[3] - box[1])

                    if len(pred_boxes) == 0:
                        # Complete miss
                        errors.append(1.0)
                        areas.append(box_area)
                        continue

                    # Compute IoU with all preds
                    ixmin = np.maximum(box[0], pred_boxes[:, 0])
                    iymin = np.maximum(box[1], pred_boxes[:, 1])
                    ixmax = np.minimum(box[2], pred_boxes[:, 2])
                    iymax = np.minimum(box[3], pred_boxes[:, 3])

                    iw = np.maximum(ixmax - ixmin, 0.0)
                    ih = np.maximum(iymax - iymin, 0.0)
                    inters = iw * ih

                    uni = (
                        (box[2] - box[0]) * (box[3] - box[1])
                        + (pred_boxes[:, 2] - pred_boxes[:, 0])
                        * (pred_boxes[:, 3] - pred_boxes[:, 1])
                        - inters
                    )

                    ious = inters / uni
                    max_iou = np.max(ious)

                    errors.append(1.0 - max_iou)
                    areas.append(box_area)

    if len(errors) > 0:
        correlation = np.corrcoef(errors, areas)[0, 1]
        print(f"Correlation between Error (1-IoU) and Box Area: {correlation:.4f}")
    else:
        print("No ground truth boxes available for failure analysis.")


def main():
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data Loading
    print("Loading datasets...")
    train_dataset, val_dataset = get_datasets(load_cached_data=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=lambda x: tuple(zip(*x)),
        pin_memory=True,
    )

    # 3. Model Initialization
    print("Initializing model...")
    model = get_model()
    model.to(device)

    # 4. Optimizer & Scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # Decay LR after 60% of epochs
    lr_decay_step = int(Config.NUM_EPOCHS * 0.6)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[lr_decay_step], gamma=Config.LR_GAMMA
    )

    # 5. Training Loop
    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")
    best_val_loss = float("inf")
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    for epoch in range(Config.NUM_EPOCHS):
        # Train
        train_metrics = train_one_epoch(
            model, optimizer, train_loader, device, epoch, print_freq=100
        )

        # Step Scheduler
        lr_scheduler.step()

        # Validate (Loss-based for checkpointing)
        val_loss = evaluate(model, val_loader, device)

        # Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

    # 6. Final Evaluation (Metric)
    print("\nLoading best model for metric evaluation...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.to(device)

    print("Calculating Final Validation Metric (mAP)...")
    final_map = calculate_map(model, val_loader, device)
    print(f"Final Validation Metric: {final_map}")

    # 7. Failure Analysis
    print("\nPerforming Failure Analysis...")
    failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.43290277912681663

    if final_map > THRESHOLD:
        print(
            f"\nMetric ({final_map}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        test_dataset = get_test_dataset(load_cached_data=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=lambda x: tuple(zip(*x)),
            pin_memory=True,
        )

        generate_submission(model, test_loader, device)
    else:
        print(
            f"\nMetric ({final_map}) <= Threshold ({THRESHOLD}). Skipping submission."
        )


if __name__ == "__main__":
    main()

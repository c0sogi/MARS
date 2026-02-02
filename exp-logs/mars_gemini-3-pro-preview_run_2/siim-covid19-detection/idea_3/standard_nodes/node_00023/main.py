import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from collections import defaultdict

# Import provided library components
from library.config import Config
from library.dataset import CovidDataset, get_transforms
from library.model import CovidMultiTaskModel
from library import utils
from library import engine

# =========================================================================
# Configuration Overrides for Fast Baseline
# =========================================================================
# Reduce epochs and training size to ensure execution within 2 hours
Config.NUM_EPOCHS = 3
Config.RPN_PRE_NMS_TOP_N_TRAIN = 1500
Config.RPN_POST_NMS_TOP_N_TRAIN = 1500
TRAIN_SUBSET_SIZE = 1500

# =========================================================================
# Helper Functions for Metric and Analysis
# =========================================================================


def calculate_iou(box1, box2):
    """
    Calculates Intersection over Union (IoU) between two boxes.
    Boxes are in [x1, y1, x2, y2] format.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def compute_ap_voc2010(recalls, precisions):
    """
    Computes Average Precision using the PASCAL VOC 2010 method (Area Under Curve).
    """
    # Prepend 0 to recalls and precisions to start from 0
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope (monotonically decreasing)
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, data_loader, device):
    """
    Evaluates the model on the validation set and computes mAP @ IoU 0.5.
    Also collects statistics for failure analysis.
    """
    model.eval()

    # Store all predictions and ground truths
    # We treat all classes 1, 2, 3 as a single "opacity" class for mAP calculation
    all_preds = []  # List of (score, image_id, box)
    gt_boxes_map = defaultdict(list)  # image_id -> list of boxes
    gt_counts = 0

    # For failure analysis: image_id -> stats
    image_stats = {}

    print("Running validation inference for mAP calculation...")
    with torch.no_grad():
        for images, targets, image_ids in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                img_id = image_ids[i]
                target = targets[i]

                # Process Ground Truth
                # target['boxes'] is on device, move to cpu
                gts = target["boxes"].cpu().numpy()
                gt_boxes_map[img_id] = gts
                gt_counts += len(gts)

                # Collect stats for failure analysis
                areas = []
                if len(gts) > 0:
                    for b in gts:
                        areas.append((b[2] - b[0]) * (b[3] - b[1]))
                    avg_area = np.mean(areas)
                else:
                    avg_area = 0.0

                image_stats[img_id] = {
                    "num_gt": len(gts),
                    "avg_area": avg_area,
                    "matched": 0,
                }

                # Process Predictions
                boxes = output["boxes"].cpu().numpy()
                scores = output["scores"].cpu().numpy()

                for b, s in zip(boxes, scores):
                    all_preds.append((s, img_id, b))

    # Sort predictions by confidence score (descending)
    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Keep track of detected GT boxes to avoid double counting (Greedy matching)
    detected_gt = {
        img_id: np.zeros(len(boxes)) for img_id, boxes in gt_boxes_map.items()
    }

    for i, (score, img_id, pred_box) in enumerate(all_preds):
        gts = gt_boxes_map[img_id]

        if len(gts) == 0:
            fp[i] = 1
            continue

        # Find best match in GT
        best_iou = 0
        best_idx = -1

        for idx, gt_box in enumerate(gts):
            iou = calculate_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_iou > 0.5:
            if detected_gt[img_id][best_idx] == 0:
                tp[i] = 1
                detected_gt[img_id][best_idx] = 1
                image_stats[img_id]["matched"] += 1
            else:
                fp[i] = 1  # Duplicate detection of same object is FP
        else:
            fp[i] = 1

    # Compute Precision and Recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    eps = 1e-6
    recalls = tp_cumsum / (gt_counts + eps)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + eps)

    ap = compute_ap_voc2010(recalls, precisions)

    return ap, image_stats


def run_failure_analysis(image_stats):
    """
    Analyzes error patterns by correlating error magnitude with input features.
    Error is defined as (1 - Recall) for images that have ground truth objects.
    """
    data = []
    for img_id, stats in image_stats.items():
        if stats["num_gt"] > 0:
            recall = stats["matched"] / stats["num_gt"]
            error = 1.0 - recall
            data.append(
                {
                    "error": error,
                    "num_gt": stats["num_gt"],
                    "avg_area": stats["avg_area"],
                }
            )

    df = pd.DataFrame(data)

    print("\n=== Failure Analysis ===")
    if len(df) > 0:
        # Calculate correlations
        corr_num = df["error"].corr(df["num_gt"])
        corr_area = df["error"].corr(df["avg_area"])

        print("Correlation between Model Error (1 - Recall) and Input Features:")
        print(f"  Num GT Boxes: {corr_num:.10f}")
        print(f"  Avg Box Area: {corr_area:.10f}")
    else:
        print("No positive samples in validation set available for failure analysis.")


# =========================================================================
# Main Execution
# =========================================================================


def main():
    utils.seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 1. Data Loading
    print("Initializing DataLoaders...")
    # Load full datasets
    train_dataset = CovidDataset(
        subset="train", transforms=get_transforms("train"), load_cached_data=True
    )
    val_dataset = CovidDataset(
        subset="val", transforms=get_transforms("val"), load_cached_data=True
    )

    # Create a subset of training data for fast baseline
    # Ensure we don't exceed dataset size
    subset_size = min(TRAIN_SUBSET_SIZE, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:subset_size]
    train_subset = Subset(train_dataset, indices)

    print(f"Training on subset of {len(train_subset)} images.")
    print(f"Validating on full set of {len(val_dataset)} images.")

    train_loader = DataLoader(
        train_subset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=utils.collate_fn,
        pin_memory=True,
    )

    # 2. Model Initialization
    print("Initializing Model...")
    model = CovidMultiTaskModel()
    model.to(device)

    # 3. Optimizer setup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    # 4. Training Loop
    print(f"Starting Training for {Config.NUM_EPOCHS} epochs...")
    best_loss = float("inf")

    for epoch in range(Config.NUM_EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{Config.NUM_EPOCHS} ---")

        # Train
        engine.train_one_epoch(model, optimizer, train_loader, device, epoch + 1)

        # Validation Loss (for checkpointing)
        val_loss = engine.evaluate_loss(model, val_loader, device)

        # Save Best Model
        if val_loss < best_loss:
            print(
                f"Validation loss improved ({best_loss:.4f} -> {val_loss:.4f}). Saving model."
            )
            best_loss = val_loss
            torch.save(
                model.state_dict(), os.path.join(Config.WORKING_DIR, "best_model.pth")
            )

    print("Training Complete.")

    # 5. Load Best Model for Evaluation
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(best_model_path):
        print("Loading best model for final evaluation...")
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Warning: Best model not found. Using current model.")

    # 6. Calculate Metric
    print("Calculating Final Validation Metric...")
    map_score, image_stats = evaluate_map(model, val_loader, device)

    # REQUIRED OUTPUT FORMAT
    print(f"Final Validation Metric: {map_score}")

    # 7. Failure Analysis
    run_failure_analysis(image_stats)

    # 8. Submission Generation
    THRESHOLD = 0.43290277912681663
    if map_score > THRESHOLD:
        print(
            f"Metric ({map_score}) > Threshold ({THRESHOLD}). Generating submission..."
        )
        engine.inference(model, device)
    else:
        print(
            f"Metric ({map_score}) <= Threshold ({THRESHOLD}). Skipping submission generation."
        )


if __name__ == "__main__":
    main()

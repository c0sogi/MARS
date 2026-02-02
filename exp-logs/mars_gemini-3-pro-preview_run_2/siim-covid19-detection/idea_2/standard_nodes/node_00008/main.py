import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import spearmanr
from torchvision.ops import box_iou

# Import from library
from library.config import Config
from library.dataset import SIIMDataset
from library.model import MultiTaskFasterRCNN
from library.engine import train_model
from library.utils import seed_everything, get_device, collate_fn
from library.inference import generate_submission


# =============================================================================
# Configuration Overrides for Fast Baseline
# =============================================================================
def configure_fast_baseline():
    """
    Adjusts configuration to run within the time limit while ensuring
    enough data is processed to get a meaningful result.
    """
    Config.MAX_TRAIN_SAMPLES = 1500  # Limit training data for speed
    Config.MAX_VAL_SAMPLES = 500  # Limit validation data for speed
    Config.NUM_EPOCHS = 3  # Few epochs for baseline
    Config.BATCH_SIZE = 8  # Fits on A100
    Config.NUM_WORKERS = 4

    # Ensure directories exist
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)


# =============================================================================
# Metric Calculation (mAP @ IoU 0.5)
# =============================================================================
def calculate_ap(recall, precision):
    """Computes the AP under the Precision-Recall curve using VOC 2010 method."""
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


@torch.no_grad()
def evaluate_map(model, data_loader, device):
    """
    Calculates mAP@0.5 for the 'opacity' class.
    Collapses all specific classes (1,2,3) to a single class for evaluation
    to match the generic 'opacity' detection requirement.
    """
    model.eval()

    all_preds = []  # List of (score, is_tp)
    num_gt_boxes = 0

    for images, targets in data_loader:
        images = list(img.to(device) for img in images)

        # Forward pass
        detections, _ = model(images)

        for i, det in enumerate(detections):
            gt_boxes = targets[i]["boxes"].to(device)
            # Filter out empty ground truths (Negative studies)
            if gt_boxes.shape[0] == 0:
                # If there are predictions on a negative image, they are FPs
                pred_scores = det["scores"].cpu().numpy()
                for score in pred_scores:
                    all_preds.append((score, 0))  # 0 = False Positive
                continue

            num_gt_boxes += gt_boxes.shape[0]

            pred_boxes = det["boxes"]
            pred_scores = det["scores"]

            if pred_boxes.shape[0] == 0:
                continue

            # Calculate IoU matrix
            iou_matrix = box_iou(pred_boxes, gt_boxes)

            # Match predictions to GT
            # Greedy matching strategy
            iou_matrix = iou_matrix.cpu().numpy()
            pred_scores = pred_scores.cpu().numpy()

            # Sort by score descending
            sorted_indices = np.argsort(-pred_scores)

            gt_matched = np.zeros(gt_boxes.shape[0], dtype=bool)

            for idx in sorted_indices:
                score = pred_scores[idx]
                ious = iou_matrix[idx]

                # Find best matching GT that hasn't been matched yet
                best_iou = -1
                best_gt_idx = -1

                for gt_idx in range(len(ious)):
                    if not gt_matched[gt_idx]:
                        if ious[gt_idx] > best_iou:
                            best_iou = ious[gt_idx]
                            best_gt_idx = gt_idx

                if best_iou >= 0.5:
                    all_preds.append((score, 1))  # True Positive
                    gt_matched[best_gt_idx] = True
                else:
                    all_preds.append((score, 0))  # False Positive

    # Compute AP
    if num_gt_boxes == 0:
        return 0.0

    if len(all_preds) == 0:
        return 0.0

    # Sort all predictions by score
    all_preds.sort(key=lambda x: x[0], reverse=True)

    tp_list = [x[1] for x in all_preds]
    fp_list = [1 - x[1] for x in all_preds]

    tp = np.cumsum(tp_list)
    fp = np.cumsum(fp_list)

    recall = tp / num_gt_boxes
    precision = tp / (tp + fp + 1e-6)

    ap = calculate_ap(recall, precision)
    return ap


# =============================================================================
# Failure Analysis
# =============================================================================
def run_failure_analysis(model, dataset, device):
    """
    Computes per-image loss and correlates it with metadata features.
    """
    print("\nRunning Failure Analysis on Validation Set...")
    model.train()  # Set to train to get loss dict

    # Use a loader with batch_size=1 to get per-image loss
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn
    )

    losses = []
    num_boxes_list = []
    box_areas_list = []

    with torch.no_grad():
        for images, targets in loader:
            images = list(img.to(device) for img in images)
            t_device = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Get Loss
            loss_dict = model(images, t_device)
            total_loss = sum(loss for loss in loss_dict.values()).item()
            losses.append(total_loss)

            # Get Features from target (CPU)
            t_cpu = targets[0]
            n_boxes = len(t_cpu["boxes"])
            num_boxes_list.append(n_boxes)

            if n_boxes > 0:
                # Calculate avg area
                boxes = t_cpu["boxes"]
                areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                avg_area = areas.mean().item()
            else:
                avg_area = 0.0
            box_areas_list.append(avg_area)

    # Compute Correlations
    corr_boxes, _ = spearmanr(losses, num_boxes_list)
    corr_area, _ = spearmanr(losses, box_areas_list)

    print(f"Correlation (Loss vs Num Boxes): {corr_boxes:.4f}")
    print(f"Correlation (Loss vs Avg Box Area): {corr_area:.4f}")


# =============================================================================
# Main Pipeline
# =============================================================================
def main():
    # 1. Setup
    configure_fast_baseline()
    seed_everything(Config.SEED)
    device = get_device()
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading Datasets...")
    train_dataset = SIIMDataset(
        split="train", debug=True
    )  # debug uses MAX_SAMPLES config
    val_dataset = SIIMDataset(split="val", debug=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # 3. Model Initialization
    print("Initializing Model...")
    model = MultiTaskFasterRCNN()
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=Config.LEARNING_RATE,
        momentum=Config.MOMENTUM,
        weight_decay=Config.WEIGHT_DECAY,
    )

    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=Config.LR_STEP_SIZE, gamma=Config.LR_GAMMA
    )

    # 4. Training
    train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        device,
        Config.NUM_EPOCHS,
    )

    # 5. Validation Metric
    print("\nCalculating Final Validation Metric (mAP@0.5)...")
    # Load best model for evaluation
    checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        print("Loaded best model checkpoint.")

    final_map = evaluate_map(model, val_loader, device)
    print(f"Final Validation Metric: {final_map}")

    # 6. Failure Analysis
    run_failure_analysis(model, val_dataset, device)

    # 7. Submission
    threshold = 0.19051633522746228
    if final_map > threshold:
        print(
            f"\nMetric ({final_map:.6f}) > Threshold ({threshold:.6f}). Generating Submission..."
        )
        generate_submission()
    else:
        print(
            f"\nMetric ({final_map:.6f}) <= Threshold ({threshold:.6f}). Skipping Submission."
        )


if __name__ == "__main__":
    main()

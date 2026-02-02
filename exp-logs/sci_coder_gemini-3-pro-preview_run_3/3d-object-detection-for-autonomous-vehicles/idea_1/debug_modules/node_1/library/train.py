import os
import time
import random
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PointPillars
from library.loss import compute_loss
from library.utils import iou_3d
from library.inference import decode_predictions, nms_process, generate_submission


def set_seed(seed=42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_image_iou_metric(pred_boxes, pred_scores, gt_boxes, thresholds):
    """
    Calculates the custom metric for a single image as defined in the task.
    Metric: Mean over thresholds of (TP / (TP + FP + FN))

    Args:
        pred_boxes: (N, 7) Tensor
        pred_scores: (N, ) Tensor
        gt_boxes: (M, 7) Tensor
        thresholds: list of float IoU thresholds
    """
    # Sort predictions by confidence
    if len(pred_scores) > 0:
        sort_idx = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sort_idx]

    num_gt = len(gt_boxes)
    num_pred = len(pred_boxes)

    # Handle edge cases for empty sets
    if num_gt == 0:
        # If no GT, any prediction is FP.
        # If pred > 0: Score = 0 / (0 + Preds + 0) = 0.
        # If pred == 0: Score = 0/0 -> Defined as 1.0 (perfect silence) or 0?
        # Task says: "If there are no ground truth objects... ANY number of predictions (false positives) will result in the image receiving a score of zero"
        if num_pred > 0:
            return 0.0
        else:
            return 1.0  # Perfect match of emptiness

    if num_pred == 0:
        # GT exists but no preds -> FN = num_gt. Score = 0.
        return 0.0

    # Pre-calculate IoU matrix (N, M)
    iou_matrix = iou_3d(pred_boxes, gt_boxes)  # (N, M)

    precisions = []

    for t in thresholds:
        tp = 0
        fp = 0

        matched_gt_indices = set()

        # Greedy matching based on confidence order
        for i in range(num_pred):
            # Get IoUs for this prediction against all GTs
            ious = iou_matrix[i]

            # Find best matching GT that hasn't been matched yet
            best_iou = -1.0
            best_gt_idx = -1

            # We iterate to find the max valid IoU
            # Note: This loop is small (M is usually < 100)
            for gt_idx in range(num_gt):
                if gt_idx in matched_gt_indices:
                    continue

                iou = ious[gt_idx].item()
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_iou > t:
                tp += 1
                matched_gt_indices.add(best_gt_idx)
            else:
                fp += 1

        fn = num_gt - len(matched_gt_indices)

        denominator = tp + fp + fn
        if denominator == 0:
            precision = 0.0
        else:
            precision = tp / denominator

        precisions.append(precision)

    return sum(precisions) / len(precisions)


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch):
    """Executes one training epoch."""
    model.train()
    total_loss = 0.0
    cls_loss_sum = 0.0
    loc_loss_sum = 0.0
    num_batches = len(loader)

    for batch in loader:
        # Move data to device
        pillars = batch["pillars"].to(device)
        coors = batch["coors"].to(device)
        n_points = batch["n_points"].to(device)
        cls_targets = batch["cls_targets"].to(device)
        reg_targets = batch["reg_targets"].to(device)
        sample_tokens = batch["sample_tokens"]

        input_dict = {
            "pillars": pillars,
            "coors": coors,
            "n_points": n_points,
            "sample_tokens": sample_tokens,
            "cls_targets": cls_targets,
            "reg_targets": reg_targets,
        }

        optimizer.zero_grad()

        output = model(input_dict)
        loss = output["loss"]

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_NORM_CLIP)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        cls_loss_sum += output["cls_loss"].item()
        loc_loss_sum += output["loc_loss"].item()

    avg_loss = total_loss / num_batches
    avg_cls = cls_loss_sum / num_batches
    avg_loc = loc_loss_sum / num_batches

    print(
        f"Epoch {epoch} Train Loss: {avg_loss:.6f} (Cls: {avg_cls:.6f}, Loc: {avg_loc:.6f})"
    )
    return avg_loss


def validate(model, loader, device, anchors):
    """Evaluates the model on the validation set using the competition metric."""
    model.eval()
    total_loss = 0.0
    metric_sum = 0.0
    num_samples = 0

    thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # Access dataset for GT parsing
    dataset = loader.dataset

    # Create a lookup for labels if not already present to speed up validation
    if not hasattr(dataset, "token_to_label"):
        dataset.token_to_label = dict(
            zip(dataset.metadata["sample_token"], dataset.metadata["label"])
        )

    with torch.no_grad():
        for batch in loader:
            pillars = batch["pillars"].to(device)
            coors = batch["coors"].to(device)
            n_points = batch["n_points"].to(device)
            sample_tokens = batch["sample_tokens"]

            # Loss Calculation inputs
            cls_targets = batch["cls_targets"].to(device)
            reg_targets = batch["reg_targets"].to(device)

            input_dict = {
                "pillars": pillars,
                "coors": coors,
                "n_points": n_points,
                "sample_tokens": sample_tokens,
                "cls_targets": cls_targets,
                "reg_targets": reg_targets,
            }

            output = model(input_dict)

            # Explicitly calculate loss for validation (since model.eval() skips it in forward)
            loss_dict = model.loss(
                output["cls_preds"], output["reg_preds"], cls_targets, reg_targets
            )
            total_loss += loss_dict["loss"].item()

            # --- Metric Calculation ---
            cls_preds = output["cls_preds"]
            reg_preds = output["reg_preds"]

            # Decode predictions
            batch_boxes, batch_scores = decode_predictions(
                cls_preds, reg_preds, anchors
            )

            for i, token in enumerate(sample_tokens):
                # 1. Get Predictions for this sample
                boxes = batch_boxes[i]
                scores = batch_scores[i]

                # Apply NMS
                final_boxes, final_scores, _ = nms_process(
                    boxes,
                    scores,
                    score_thresh=Config.NMS_SCORE_THRESHOLD,
                    iou_thresh=Config.NMS_IOU_THRESHOLD,
                    max_proposals=Config.MAX_PROPOSALS,
                )

                if final_boxes is None:
                    final_boxes = torch.empty((0, 7), device=device)
                    final_scores = torch.empty((0,), device=device)

                # 2. Get Ground Truth for this sample
                label_str = dataset.token_to_label.get(token, "")
                gt_boxes_np = dataset._parse_labels(label_str)

                if len(gt_boxes_np) > 0:
                    gt_boxes = torch.from_numpy(gt_boxes_np[:, :7]).to(device)
                else:
                    gt_boxes = torch.empty((0, 7), device=device)

                # 3. Calculate Metric
                score = calculate_image_iou_metric(
                    final_boxes, final_scores, gt_boxes, thresholds
                )
                metric_sum += score
                num_samples += 1

    avg_loss = total_loss / len(loader)
    avg_metric = metric_sum / num_samples if num_samples > 0 else 0.0

    print(f"Validation Loss: {avg_loss:.10f}")
    print(f"Validation Metric (mAP): {avg_metric:.10f}")

    return avg_loss, avg_metric


def train_model(epochs=Config.EPOCHS, batch_size=Config.BATCH_SIZE, max_samples=None):
    """
    Main function to train the PointPillars model.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    print(f"Starting training on {device}...")

    # Initialize Datasets
    train_dataset = NuScenesDataset(split="train", load_cached_data=True)
    val_dataset = NuScenesDataset(split="val", load_cached_data=True)

    # Optional subsampling for debugging
    if max_samples:
        train_dataset.metadata = train_dataset.metadata.iloc[:max_samples]
        val_dataset.metadata = val_dataset.metadata.iloc[:max_samples]
        print(f"Subsampled datasets to {max_samples} samples.")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
        pin_memory=True,
    )

    # Model
    model = PointPillars().to(device)

    # Optimization
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,
        div_factor=10,
        final_div_factor=100,
    )

    # Anchors (needed for validation decoding)
    anchors = torch.from_numpy(val_dataset.anchors).to(device)

    # Training Loop
    best_metric = -1.0
    patience = 3
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device, epoch
        )
        val_loss, val_metric = validate(model, val_loader, device, anchors)

        # Checkpoint & Early Stopping
        if val_metric > best_metric:
            best_metric = val_metric
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric": best_metric,
                },
                Config.MODEL_SAVE_PATH,
            )
            print(f"New best model saved with mAP: {best_metric:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print("Training complete.")

    # Generate Submission
    if os.path.exists(Config.MODEL_SAVE_PATH):
        generate_submission(model_path=Config.MODEL_SAVE_PATH)

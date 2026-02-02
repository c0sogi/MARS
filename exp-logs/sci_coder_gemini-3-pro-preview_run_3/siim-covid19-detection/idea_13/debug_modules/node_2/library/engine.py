import os
import time
import math
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    AverageMeter,
    box_cxcywh_to_xyxy,
    box_iou,
    format_prediction_string,
    seed_everything,
)
from library.dataset import SIIMDataset, collate_fn, STUDY_CLASSES
from library.model import build_model


def restore_coordinates(
    boxes: torch.Tensor, orig_h: int, orig_w: int, img_size: int
) -> torch.Tensor:
    """
    Maps normalized boxes (cx, cy, w, h) in [0, 1] relative to img_size
    back to absolute coordinates (x1, y1, x2, y2) in the original image.
    Handles the Letterbox resizing (LongestMaxSize + Pad).
    """
    # 1. Convert to absolute coordinates in the transformed image (img_size x img_size)
    # boxes is (N, 4) in cx, cy, w, h
    boxes_xyxy = box_cxcywh_to_xyxy(boxes)  # (N, 4) x1, y1, x2, y2 normalized
    boxes_abs = boxes_xyxy * img_size

    # 2. Determine padding and scale used during preprocessing
    if orig_h > orig_w:
        scale = img_size / orig_h
        new_w = orig_w * scale
        pad_x = (img_size - new_w) / 2
        pad_y = 0
    else:
        scale = img_size / orig_w
        new_h = orig_h * scale
        pad_x = 0
        pad_y = (img_size - new_h) / 2

    # 3. Remove padding
    boxes_abs[:, [0, 2]] -= pad_x
    boxes_abs[:, [1, 3]] -= pad_y

    # 4. Rescale to original size
    boxes_orig = boxes_abs / scale

    # Clip to image boundaries
    boxes_orig[:, [0, 2]] = boxes_orig[:, [0, 2]].clamp(min=0, max=orig_w)
    boxes_orig[:, [1, 3]] = boxes_orig[:, [1, 3]].clamp(min=0, max=orig_h)

    return boxes_orig


def calculate_map(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculates mAP@IoU>0.5 for a single class (Opacity).
    """
    tp = []
    fp = []
    scores = []
    num_gt = 0

    for pred_boxes, pred_scores, gt_boxes in zip(
        pred_boxes_list, pred_scores_list, gt_boxes_list
    ):
        num_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by score
        sorted_indices = torch.argsort(pred_scores, descending=True)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        scores.append(pred_scores.cpu().numpy())

        if len(gt_boxes) == 0:
            tp.append(np.zeros(len(pred_boxes)))
            fp.append(np.ones(len(pred_boxes)))
            continue

        # Calculate IoU
        # pred_boxes and gt_boxes should be in same format.
        # Here we assume both are (x1, y1, x2, y2) absolute or both normalized.
        # We use normalized for metric calculation to avoid scaling issues.
        ious, _ = box_iou(pred_boxes, gt_boxes)  # (N_pred, N_gt)

        assigned_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
        img_tp = np.zeros(len(pred_boxes))
        img_fp = np.zeros(len(pred_boxes))

        for i in range(len(pred_boxes)):
            max_iou, max_idx = torch.max(ious[i], dim=0)
            if max_iou > iou_threshold:
                if not assigned_gt[max_idx]:
                    img_tp[i] = 1
                    assigned_gt[max_idx] = True
                else:
                    img_fp[i] = 1
            else:
                img_fp[i] = 1

        tp.append(img_tp)
        fp.append(img_fp)

    if num_gt == 0:
        return 0.0

    if not tp:
        return 0.0

    tp = np.concatenate(tp)
    fp = np.concatenate(fp)
    scores = np.concatenate(scores)

    # Sort by score across all images
    indices = np.argsort(-scores)
    tp = tp[indices]
    fp = fp[indices]

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    precision = cum_tp / (cum_tp + cum_fp + 1e-8)
    recall = cum_tp / num_gt

    # Compute AP using 11-point interpolation or simple area
    # Using simple area under PR curve
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recall >= t) == 0:
            p = 0
        else:
            p = np.max(precision[recall >= t])
        ap += p / 11.0

    return ap


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    accumulation_steps: int = 1,
) -> Dict[str, float]:
    model.train()
    criterion.train()

    loss_meter = AverageMeter()
    loss_study_meter = AverageMeter()
    loss_box_meter = AverageMeter()

    optimizer.zero_grad()

    start_time = time.time()

    for i, (samples, targets) in enumerate(dataloader):
        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(samples, targets)
        loss_dict = criterion(outputs, targets)

        # Weighted sum of losses
        weight_dict = criterion.detection_criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )

        # Add study loss (handled separately in MultiTaskCriterion wrapper)
        if "loss_study" in loss_dict:
            losses += loss_dict["loss_study"]

        # Normalize for gradient accumulation
        losses = losses / accumulation_steps
        losses.backward()

        if (i + 1) % accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_MAX_NORM)
            optimizer.step()
            optimizer.zero_grad()

        # Logging
        current_loss = losses.item() * accumulation_steps
        loss_meter.update(current_loss, samples.size(0))

        if "loss_study" in loss_dict:
            loss_study_meter.update(loss_dict["loss_study"].item(), samples.size(0))

        if "loss_bbox" in loss_dict:
            loss_box_meter.update(loss_dict["loss_bbox"].item(), samples.size(0))

    epoch_time = time.time() - start_time
    print(
        f"Epoch [{epoch}] Train Loss: {loss_meter.avg:.4f} "
        f"(Study: {loss_study_meter.avg:.4f}, Box: {loss_box_meter.avg:.4f}) "
        f"Time: {epoch_time:.1f}s"
    )

    return {"loss": loss_meter.avg}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    criterion.eval()

    loss_meter = AverageMeter()

    # Metrics storage
    pred_boxes_list = []
    pred_scores_list = []
    gt_boxes_list = []

    study_correct = 0
    study_total = 0

    for samples, targets in dataloader:
        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        outputs = model(samples)
        loss_dict = criterion(outputs, targets)

        # Loss calculation
        weight_dict = criterion.detection_criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        if "loss_study" in loss_dict:
            losses += loss_dict["loss_study"]

        loss_meter.update(losses.item(), samples.size(0))

        # Study Accuracy
        if "study_logits" in outputs:
            study_preds = torch.argmax(outputs["study_logits"], dim=1)
            study_gts = torch.stack([t["study_label"] for t in targets])
            study_correct += (study_preds == study_gts).sum().item()
            study_total += samples.size(0)

        # Detection mAP Prep
        # Get boxes and scores from the last layer of decoder
        pred_logits = outputs["pred_logits"]  # (B, NQ, 1)
        pred_boxes = outputs["pred_boxes"]  # (B, NQ, 4) cxcywh norm

        prob = pred_logits.sigmoid().squeeze(-1)  # (B, NQ)

        # Convert preds to xyxy normalized for IoU calc
        pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)

        for i in range(len(targets)):
            # Filter by confidence for metric calculation speedup
            keep = prob[i] > Config.CONFIDENCE_THRESHOLD

            p_boxes = pred_boxes_xyxy[i][keep]
            p_scores = prob[i][keep]

            gt_b = targets[i]["boxes"]  # (M, 4) cxcywh norm
            gt_b_xyxy = box_cxcywh_to_xyxy(gt_b)

            pred_boxes_list.append(p_boxes)
            pred_scores_list.append(p_scores)
            gt_boxes_list.append(gt_b_xyxy)

    # Compute Metrics
    map_score = calculate_map(pred_boxes_list, pred_scores_list, gt_boxes_list)
    study_acc = study_correct / study_total if study_total > 0 else 0.0

    print(
        f"Validation Loss: {loss_meter.avg:.4f} | "
        f"mAP@0.5: {map_score:.4f} | "
        f"Study Acc: {study_acc:.4f}"
    )

    return {"val_loss": loss_meter.avg, "map": map_score, "study_acc": study_acc}


def train(
    debug: bool = False, epochs: int = Config.EPOCHS, save_path: str = "best_model.pth"
):
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Data
    train_dataset = SIIMDataset("train", debug=debug)
    val_dataset = SIIMDataset("val", debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model
    model, criterion = build_model(Config)
    model.to(device)
    criterion.to(device)

    # 3. Optimizer
    # Separate parameter groups for backbone
    param_dicts = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" not in n and p.requires_grad
            ],
            "lr": Config.LEARNING_RATE,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": Config.BACKBONE_LR,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    # 4. Loop
    best_map = 0.0
    patience_counter = 0

    save_file = os.path.join(Config.WORKING_DIR, save_path)

    print(f"Starting training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            criterion,
            train_loader,
            optimizer,
            device,
            epoch,
            Config.ACCUMULATE_GRAD_BATCHES,
        )

        val_stats = evaluate(model, criterion, val_loader, device)

        scheduler.step()

        # Checkpointing
        if val_stats["map"] > best_map:
            best_map = val_stats["map"]
            patience_counter = 0
            torch.save(model.state_dict(), save_file)
            print(f"New best mAP: {best_map:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best mAP: {best_map:.4f}")


def inference(model_path: str = "best_model.pth"):
    device = torch.device(Config.DEVICE)

    # Load Model
    model, _ = build_model(Config)

    # Load Weights
    full_path = os.path.join(Config.WORKING_DIR, model_path)
    if os.path.exists(full_path):
        state_dict = torch.load(full_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded model from {full_path}")
    else:
        print(f"Warning: Model file {full_path} not found. Using random weights.")

    model.to(device)
    model.eval()

    # Load Test Data
    test_dataset = SIIMDataset("test", load_cached_data=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    results = []

    print("Running inference...")
    with torch.no_grad():
        for samples, targets in test_loader:
            samples = samples.to(device)
            # targets contains 'study_id', 'image_id', 'orig_size'

            outputs = model(samples)

            # 1. Study Predictions
            study_logits = outputs["study_logits"]
            study_probs = torch.softmax(study_logits, dim=1)
            study_preds = torch.argmax(study_probs, dim=1)  # (B,)

            # 2. Box Predictions
            pred_logits = outputs["pred_logits"]  # (B, NQ, 1)
            pred_boxes = outputs["pred_boxes"]  # (B, NQ, 4) cxcywh norm
            pred_scores = pred_logits.sigmoid().squeeze(-1)  # (B, NQ)

            for i in range(len(samples)):
                study_id = targets[i]["study_id"]
                image_id = targets[i]["image_id"]
                orig_h, orig_w = targets[i]["orig_size"]

                # Study Result
                study_cls_idx = study_preds[i].item()
                study_label_str = STUDY_CLASSES[study_cls_idx]

                # Map full string to required submission format
                # 'Negative for Pneumonia' -> 'negative'
                # 'Typical Appearance' -> 'typical'
                # 'Indeterminate Appearance' -> 'indeterminate'
                # 'Atypical Appearance' -> 'atypical'
                study_map = {
                    "Negative for Pneumonia": "negative",
                    "Typical Appearance": "typical",
                    "Indeterminate Appearance": "indeterminate",
                    "Atypical Appearance": "atypical",
                }
                short_label = study_map.get(study_label_str, "negative")

                # Study Prediction String: "class confidence 0 0 1 1"
                # We use confidence 1.0 for the chosen class or the softmax prob
                conf = study_probs[i, study_cls_idx].item()
                study_pred_str = f"{short_label} {conf:.6f} 0 0 1 1"

                results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_pred_str}
                )

                # Image Result
                if short_label == "negative":
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    # Filter boxes
                    keep = pred_scores[i] > Config.CONFIDENCE_THRESHOLD
                    valid_scores = pred_scores[i][keep]
                    valid_boxes = pred_boxes[i][keep]  # (M, 4) cxcywh norm

                    if len(valid_boxes) == 0:
                        image_pred_str = "none 1 0 0 1 1"
                    else:
                        # Restore coordinates to original image size
                        restored_boxes = restore_coordinates(
                            valid_boxes, orig_h.item(), orig_w.item(), Config.IMG_SIZE
                        )

                        # Format: "opacity score xmin ymin xmax ymax ..."
                        labels_list = ["opacity"] * len(valid_boxes)
                        # restored_boxes is (M, 4) x1y1x2y2
                        boxes_list = restored_boxes.cpu().numpy().tolist()
                        scores_list = valid_scores.cpu().numpy().tolist()

                        image_pred_str = format_prediction_string(
                            labels_list, boxes_list, scores_list
                        )

                results.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

    # Save Submission
    df_sub = pd.DataFrame(results)
    sub_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")

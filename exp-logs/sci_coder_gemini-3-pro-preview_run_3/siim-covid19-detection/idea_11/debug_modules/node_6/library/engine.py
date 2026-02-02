import torch
import math
import sys
import numpy as np
from typing import Iterable, Dict, List
import torch.nn.functional as F

from library.config import Config
from library.utils import box_cxcywh_to_xyxy, box_iou, scale_coords

# Default weights for DINO-like training
LOSS_WEIGHTS = {
    "loss_ce": 1.0,
    "loss_bbox": 5.0,
    "loss_giou": 2.0,
    "loss_study": 1.0,
}


def get_loss_weights():
    """
    Generates a dictionary of weights for all loss components including
    auxiliary, DN, and encoder outputs.
    """
    weights = LOSS_WEIGHTS.copy()

    # Add weights for auxiliary outputs (same as main)
    for i in range(Config.DEC_LAYERS - 1):
        weights[f"loss_ce_{i}"] = LOSS_WEIGHTS["loss_ce"]
        weights[f"loss_bbox_{i}"] = LOSS_WEIGHTS["loss_bbox"]
        weights[f"loss_giou_{i}"] = LOSS_WEIGHTS["loss_giou"]

    # Add weights for DN (DeNoising) outputs
    weights["loss_ce_dn"] = LOSS_WEIGHTS["loss_ce"]
    weights["loss_bbox_dn"] = LOSS_WEIGHTS["loss_bbox"]
    weights["loss_giou_dn"] = LOSS_WEIGHTS["loss_giou"]

    # Add weights for Encoder outputs (Two-stage selection)
    weights["loss_ce_enc"] = LOSS_WEIGHTS["loss_ce"]
    weights["loss_bbox_enc"] = LOSS_WEIGHTS["loss_bbox"]
    weights["loss_giou_enc"] = LOSS_WEIGHTS["loss_giou"]

    return weights


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0.1,
):
    model.train()
    criterion.train()

    weights = get_loss_weights()
    total_loss_val = 0.0
    steps = 0

    # Iterate over data loader
    # Note: data_loader yields (samples, targets)
    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        # We pass targets to model for Contrastive DeNoising (CDN)
        outputs = model(samples, targets)

        # Calculate losses
        loss_dict = criterion(outputs, targets)

        # Weighted sum
        losses = sum(
            loss_dict[k] * weights.get(k, 1.0) for k in loss_dict.keys() if k in weights
        )

        # Backprop
        optimizer.zero_grad()
        losses.backward()

        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        total_loss_val += losses.item()
        steps += 1

    avg_loss = total_loss_val / max(steps, 1)
    print(f"Epoch: {epoch} | Train Loss: {avg_loss:.4f}")
    return avg_loss


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    device: torch.device,
):
    model.eval()
    criterion.eval()

    weights = get_loss_weights()
    total_loss = 0.0
    steps = 0

    # For mAP calculation
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    # For Study Accuracy
    study_preds = []
    study_gts = []

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass (no targets for CDN during eval)
        outputs = model(samples)

        # Calculate validation loss
        loss_dict = criterion(outputs, targets)
        losses = sum(
            loss_dict[k] * weights.get(k, 1.0) for k in loss_dict.keys() if k in weights
        )
        total_loss += losses.item()
        steps += 1

        # --- Process Predictions for Metrics ---

        # 1. Study Level
        if "study_logits" in outputs:
            s_logits = outputs["study_logits"]
            s_preds = torch.argmax(s_logits, dim=1)
            s_gts = torch.stack([t["study_label"] for t in targets])

            study_preds.extend(s_preds.cpu().numpy())
            study_gts.extend(s_gts.cpu().numpy())

        # 2. Image Level (Detection)
        # outputs["pred_logits"]: (B, Q, NumClasses)
        # outputs["pred_boxes"]: (B, Q, 4) in (cx, cy, w, h) normalized

        prob = outputs["pred_logits"].sigmoid()
        # We only have 1 class (opacity), index 0
        scores = prob[:, :, 0]
        boxes = outputs["pred_boxes"]

        # Convert boxes to xyxy
        boxes_xyxy = box_cxcywh_to_xyxy(boxes)

        batch_size = scores.shape[0]
        for i in range(batch_size):
            # Get image dimensions for scaling
            img_h, img_w = targets[i]["orig_size"].cpu().numpy()

            # Scale boxes to original image size
            # The model outputs normalized coords (0-1).
            # We multiply by IMG_SIZE (1024) to get coords in resized space,
            # then we need to reverse the letterbox.
            # However, `scale_coords` in utils expects coords in the resized image space.

            # 1. Denormalize to Config.IMG_SIZE
            current_boxes = boxes_xyxy[i].clone()
            current_boxes[:, 0::2] *= Config.IMG_SIZE
            current_boxes[:, 1::2] *= Config.IMG_SIZE

            # 2. Reverse Letterbox
            # We need ratio and pad. We can re-derive them or store them.
            # `dataset.py` stores `orig_size` and `img_size`.
            # ratio = min(1024/h, 1024/w)

            target_size = Config.IMG_SIZE
            orig_h, orig_w = float(img_h), float(img_w)
            r = min(target_size / orig_h, target_size / orig_w)
            pad_w = (target_size - orig_w * r) / 2
            pad_h = (target_size - orig_h * r) / 2

            # Use utility to scale back
            # scale_coords(coords, ratio, pad, to_original=True)
            final_boxes = scale_coords(
                current_boxes, ratio=r, pad=(pad_w, pad_h), to_original=True
            )

            # Filter by score threshold for mAP calculation efficiency?
            # Standard mAP takes all, but we can clip very low ones.
            keep = scores[i] > 0.001

            pred_boxes_list.append(final_boxes[keep].cpu())
            pred_scores_list.append(scores[i][keep].cpu())
            # All predicted class 0
            pred_labels_list.append(
                torch.zeros_like(scores[i][keep], dtype=torch.long).cpu()
            )

            # GT
            gt_boxes_raw = targets[i]["boxes"].cpu()  # Normalized cxcywh
            if len(gt_boxes_raw) > 0:
                gt_xyxy = box_cxcywh_to_xyxy(gt_boxes_raw)
                gt_xyxy[:, 0::2] *= Config.IMG_SIZE
                gt_xyxy[:, 1::2] *= Config.IMG_SIZE

                final_gt = scale_coords(
                    gt_xyxy, ratio=r, pad=(pad_w, pad_h), to_original=True
                )
                gt_boxes_list.append(final_gt)
                gt_labels_list.append(targets[i]["labels"].cpu())
            else:
                gt_boxes_list.append(torch.tensor([], dtype=torch.float32))
                gt_labels_list.append(torch.tensor([], dtype=torch.long))

    avg_loss = total_loss / max(steps, 1)

    # Calculate Metrics
    map_50 = calculate_map(
        pred_boxes_list, pred_scores_list, gt_boxes_list, iou_threshold=0.5
    )

    study_acc = 0.0
    if len(study_gts) > 0:
        study_acc = np.mean(np.array(study_preds) == np.array(study_gts))

    print(
        f"Eval Loss: {avg_loss:.4f} | mAP@0.5: {map_50:.10f} | Study Acc: {study_acc:.10f}"
    )

    return {"loss": avg_loss, "map_50": map_50, "study_acc": study_acc}


def calculate_map(
    pred_boxes_list: List[torch.Tensor],
    pred_scores_list: List[torch.Tensor],
    gt_boxes_list: List[torch.Tensor],
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculates mAP@IoU for a single class (Opacity).
    Uses 11-point interpolation or standard AUC.
    """
    # Flatten everything
    all_preds = []  # (score, image_idx, box_idx)

    total_gt = 0

    for i in range(len(pred_scores_list)):
        scores = pred_scores_list[i]
        boxes = pred_boxes_list[i]

        # Add to list
        for j in range(len(scores)):
            all_preds.append({"score": float(scores[j]), "box": boxes[j], "img_idx": i})

        total_gt += len(gt_boxes_list[i])

    if total_gt == 0:
        return 0.0 if len(all_preds) > 0 else 0.0  # Undefined, usually 0

    # Sort predictions by score descending
    all_preds.sort(key=lambda x: x["score"], reverse=True)

    tp = np.zeros(len(all_preds))
    fp = np.zeros(len(all_preds))

    # Track which GTs have been matched
    gt_matched = [torch.zeros(len(gts), dtype=torch.bool) for gts in gt_boxes_list]

    for i, pred in enumerate(all_preds):
        img_idx = pred["img_idx"]
        pred_box = pred["box"].unsqueeze(0)  # (1, 4)

        gts = gt_boxes_list[img_idx]  # (M, 4)

        if len(gts) == 0:
            fp[i] = 1
            continue

        # Compute IoU
        # box_iou expects (N, 4) and (M, 4)
        ious = box_iou(pred_box, gts)[0]  # (M,)

        if len(ious) > 0:
            max_iou, max_idx = torch.max(ious, dim=0)

            if max_iou >= iou_threshold:
                if not gt_matched[img_idx][max_idx]:
                    tp[i] = 1
                    gt_matched[img_idx][max_idx] = True
                else:
                    fp[i] = 1  # Duplicate detection
            else:
                fp[i] = 1
        else:
            fp[i] = 1

    # Compute Precision and Recall
    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)

    recalls = cum_tp / total_gt
    precisions = cum_tp / (cum_tp + cum_fp + 1e-8)

    # Compute AP (Area Under Curve)
    # We use the continuous integration method (standard in modern VOC/COCO)
    # Append sentinels
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate area under curve
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

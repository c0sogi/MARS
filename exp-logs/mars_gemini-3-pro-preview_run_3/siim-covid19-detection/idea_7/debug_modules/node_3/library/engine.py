import torch
import numpy as np
import pandas as pd
import sys
import os
import time
import math
from library.config import Config
from library.utils import (
    bb_intersection_over_union,
    weighted_boxes_fusion,
    format_prediction_string,
)


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()

    losses_tracker = {}
    header = f"Epoch: [{epoch}]"

    for step, (images, targets) in enumerate(data_loader):
        images = images.to(device)
        # targets is a list of dicts, move tensors to device
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)
        optimizer.step()

        # Track losses
        loss_dict_reduced = {k: v.item() for k, v in loss_dict.items()}
        loss_dict_reduced["total_loss"] = losses.item()

        for k, v in loss_dict_reduced.items():
            if k not in losses_tracker:
                losses_tracker[k] = []
            losses_tracker[k].append(v)

        if step % print_freq == 0:
            loss_str = " ".join(
                [f"{k}: {np.mean(v):.6f}" for k, v in losses_tracker.items()]
            )
            print(f"{header} Step: [{step}/{len(data_loader)}] {loss_str}")

    # Return average losses
    avg_losses = {k: np.mean(v) for k, v in losses_tracker.items()}
    return avg_losses


def calculate_map(pred_boxes_list, pred_scores_list, gt_boxes_list, iou_threshold=0.5):
    """
    Calculates mAP@IoU for a single class (opacity) using VOC 2010 method.
    """
    tp = []
    fp = []
    scores = []
    num_gt = 0

    for i in range(len(gt_boxes_list)):
        gt_boxes = gt_boxes_list[i]
        pred_boxes = pred_boxes_list[i]
        pred_scores = pred_scores_list[i]

        num_gt += len(gt_boxes)

        if len(pred_boxes) == 0:
            continue

        # Sort predictions by score
        sorted_indices = np.argsort(-pred_scores)
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        gt_matched = np.zeros(len(gt_boxes))

        for b_idx, box in enumerate(pred_boxes):
            scores.append(pred_scores[b_idx])

            if len(gt_boxes) == 0:
                tp.append(0)
                fp.append(1)
                continue

            # Calculate IoU with all GT boxes
            ious = np.array(
                [bb_intersection_over_union(box, gt_box) for gt_box in gt_boxes]
            )
            max_iou = np.max(ious) if len(ious) > 0 else 0
            max_idx = np.argmax(ious) if len(ious) > 0 else -1

            if max_iou >= iou_threshold:
                if gt_matched[max_idx] == 0:
                    tp.append(1)
                    fp.append(0)
                    gt_matched[max_idx] = 1
                else:
                    tp.append(0)
                    fp.append(1)
            else:
                tp.append(0)
                fp.append(1)

    if num_gt == 0:
        return 0.0

    if len(scores) == 0:
        return 0.0

    # Convert to numpy
    tp = np.array(tp)
    fp = np.array(fp)
    scores = np.array(scores)

    # Sort by score descending
    sorted_indices = np.argsort(-scores)
    tp = tp[sorted_indices]
    fp = fp[sorted_indices]

    # Compute Cumulative
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)

    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

    # Compute AP (VOC 2010)
    # Append sentinel values
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # Integrate
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap


def evaluate(model, data_loader, device):
    model.eval()

    pred_boxes_all = []
    pred_scores_all = []
    gt_boxes_all = []

    study_correct = 0
    study_total = 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            # targets is list of dicts

            # Forward
            detections, study_probs = model(images)

            # Process Study Metrics
            pred_labels = torch.argmax(study_probs, dim=1).cpu()
            gt_labels = torch.stack([t["study_label"] for t in targets]).cpu()

            study_correct += (pred_labels == gt_labels).sum().item()
            study_total += len(gt_labels)

            # Process Detection Metrics
            for i, det in enumerate(detections):
                # Predictions (already on cpu via model postprocess usually, but let's ensure)
                p_boxes = det["boxes"].cpu().numpy()
                p_scores = det["scores"].cpu().numpy()

                # Ground Truth
                t_boxes = targets[i]["boxes"].cpu().numpy()

                pred_boxes_all.append(p_boxes)
                pred_scores_all.append(p_scores)
                gt_boxes_all.append(t_boxes)

    # Calculate Accuracy
    study_acc = study_correct / study_total if study_total > 0 else 0.0

    # Calculate mAP
    map_score = calculate_map(
        pred_boxes_all, pred_scores_all, gt_boxes_all, iou_threshold=0.5
    )

    return {"study_accuracy": study_acc, "map_0.5": map_score}


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    checkpoint_path,
):
    best_score = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # Train
        train_metrics = train_one_epoch(model, optimizer, train_loader, device, epoch)
        print(f"Train Loss: {train_metrics['total_loss']:.6f}")

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        print(f"Val Study Acc: {val_metrics['study_accuracy']:.6f}")
        print(f"Val mAP@0.5: {val_metrics['map_0.5']:.6f}")

        # Scheduler Step
        scheduler.step()

        # Score calculation (Use mAP for early stopping)
        current_score = val_metrics["map_0.5"]

        if current_score > best_score:
            best_score = current_score
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"New best model saved with mAP: {best_score:.6f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break


def inference(model, data_loader, device, submission_path):
    model.eval()

    results = []

    # Mapping for study labels
    study_map = {
        "Negative for Pneumonia": "negative",
        "Typical Appearance": "typical",
        "Indeterminate Appearance": "indeterminate",
        "Atypical Appearance": "atypical",
    }

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)

            # 1. Original Prediction
            dets_orig, study_orig = model(images)

            # 2. TTA (Horizontal Flip)
            if Config.USE_TTA:
                images_flipped = torch.flip(images, [3])  # Flip width (dim 3)
                dets_flip, study_flip = model(images_flipped)

            # Process batch
            batch_size = images.shape[0]
            for i in range(batch_size):
                image_id = targets[i]["id_str"]
                study_id = targets[i]["study_id_str"]

                # Retrieve original dimensions for rescaling
                orig_h = targets[i]["original_size"][0].item()
                orig_w = targets[i]["original_size"][1].item()

                # --- Study Prediction ---
                if Config.USE_TTA:
                    s_prob = (study_orig[i] + study_flip[i]) / 2.0
                else:
                    s_prob = study_orig[i]

                study_label_idx = torch.argmax(s_prob).item()
                study_label_full = Config.STUDY_LABELS[study_label_idx]
                short_label = study_map[study_label_full]

                # Format: "label confidence 0 0 1 1"
                study_pred_str = f"{short_label} {s_prob[study_label_idx]:.4f} 0 0 1 1"
                results.append(
                    {"id": f"{study_id}_study", "PredictionString": study_pred_str}
                )

                # --- Image Prediction ---
                # If Negative, force none
                if study_label_idx == Config.NEGATIVE_CLASS_IDX:
                    image_pred_str = "none 1 0 0 1 1"
                else:
                    # Gather boxes for WBF
                    boxes_list = []
                    scores_list = []
                    labels_list = []

                    # Original
                    b_orig = dets_orig[i]["boxes"].cpu().numpy()
                    s_orig = dets_orig[i]["scores"].cpu().numpy()
                    l_orig = dets_orig[i]["labels"].cpu().numpy()

                    boxes_list.append(b_orig)
                    scores_list.append(s_orig)
                    labels_list.append(l_orig)

                    if Config.USE_TTA:
                        b_flip = dets_flip[i]["boxes"].cpu().numpy()
                        s_flip = dets_flip[i]["scores"].cpu().numpy()
                        l_flip = dets_flip[i]["labels"].cpu().numpy()

                        # Un-flip boxes in 800x800 space
                        if len(b_flip) > 0:
                            b_flip_fixed = b_flip.copy()
                            # [x1, y1, x2, y2]
                            # Flip width: new_x1 = width - old_x2
                            b_flip_fixed[:, 0] = Config.IMG_SIZE - b_flip[:, 2]
                            b_flip_fixed[:, 2] = Config.IMG_SIZE - b_flip[:, 0]

                            boxes_list.append(b_flip_fixed)
                            scores_list.append(s_flip)
                            labels_list.append(l_flip)

                    # WBF
                    boxes, scores, labels = weighted_boxes_fusion(
                        boxes_list,
                        scores_list,
                        labels_list,
                        iou_thr=Config.WBF_IOU_THRESHOLD,
                        skip_box_thr=Config.WBF_CONF_THRESHOLD,
                    )

                    # Rescale boxes back to original DICOM size
                    # Reverse Letterbox: (val - pad) / scale
                    scale = min(Config.IMG_SIZE / orig_h, Config.IMG_SIZE / orig_w)
                    new_h, new_w = int(orig_h * scale), int(orig_w * scale)
                    pad_h = (Config.IMG_SIZE - new_h) // 2
                    pad_w = (Config.IMG_SIZE - new_w) // 2

                    if len(boxes) > 0:
                        boxes[:, 0] = (boxes[:, 0] - pad_w) / scale
                        boxes[:, 2] = (boxes[:, 2] - pad_w) / scale
                        boxes[:, 1] = (boxes[:, 1] - pad_h) / scale
                        boxes[:, 3] = (boxes[:, 3] - pad_h) / scale

                        # Clip
                        boxes[:, 0] = np.clip(boxes[:, 0], 0, orig_w)
                        boxes[:, 2] = np.clip(boxes[:, 2], 0, orig_w)
                        boxes[:, 1] = np.clip(boxes[:, 1], 0, orig_h)
                        boxes[:, 3] = np.clip(boxes[:, 3], 0, orig_h)

                    image_pred_str = format_prediction_string(labels, boxes, scores)

                results.append(
                    {"id": f"{image_id}_image", "PredictionString": image_pred_str}
                )

    # Create DataFrame
    df_res = pd.DataFrame(results)

    # Drop duplicates (keeping first occurrence)
    df_res = df_res.drop_duplicates(subset=["id"])

    df_res.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

import math
import sys
import torch
from typing import Iterable

from library.config import Config
from library.utils import AverageMeter, calculate_map, box_cxcywh_to_xyxy


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
):
    """
    Executes one epoch of training.
    """
    model.train()
    criterion.train()

    # Initialize meters for tracking losses
    loss_meter = AverageMeter()
    loss_ce_meter = AverageMeter()
    loss_bbox_meter = AverageMeter()
    loss_giou_meter = AverageMeter()
    loss_study_meter = AverageMeter()

    print(f"Epoch: [{epoch}] Starting training loop...")

    for i, (samples, targets) in enumerate(data_loader):
        # Move inputs to device
        samples = samples.to(device)
        # Targets is a list of dicts, move tensors inside to device
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # Forward pass
        # pixel_mask is None because we use Letterbox resizing (fixed size batch)
        outputs = model(samples)

        # Compute losses
        loss_dict = criterion(outputs, targets)
        losses = loss_dict["loss"]

        # Backward pass
        optimizer.zero_grad()
        losses.backward()

        # Gradient Clipping
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()

        # Update logs
        batch_size = samples.shape[0]
        loss_meter.update(losses.item(), batch_size)

        if "loss_ce" in loss_dict:
            loss_ce_meter.update(loss_dict["loss_ce"].item(), batch_size)
        if "loss_bbox" in loss_dict:
            loss_bbox_meter.update(loss_dict["loss_bbox"].item(), batch_size)
        if "loss_giou" in loss_dict:
            loss_giou_meter.update(loss_dict["loss_giou"].item(), batch_size)
        if "loss_study" in loss_dict:
            loss_study_meter.update(loss_dict["loss_study"].item(), batch_size)

        # Logging
        if i % 10 == 0:
            print(
                f"Epoch: [{epoch}] Step: [{i}/{len(data_loader)}] "
                f"Loss: {loss_meter.val:.4f} ({loss_meter.avg:.4f}) "
                f"Study Loss: {loss_study_meter.val:.4f} ({loss_study_meter.avg:.4f})"
            )

    return {
        "loss": loss_meter.avg,
        "loss_ce": loss_ce_meter.avg,
        "loss_bbox": loss_bbox_meter.avg,
        "loss_giou": loss_giou_meter.avg,
        "loss_study": loss_study_meter.avg,
    }


@torch.no_grad()
def evaluate(model, criterion, data_loader, device):
    """
    Evaluates the model on the validation set.
    Computes Loss, Study Accuracy, and mAP.
    """
    model.eval()
    criterion.eval()

    loss_meter = AverageMeter()
    study_acc_meter = AverageMeter()

    # Lists to store data for mAP calculation
    pred_boxes_list = []
    pred_scores_list = []
    pred_labels_list = []
    gt_boxes_list = []
    gt_labels_list = []

    print("Starting evaluation...")

    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [
            {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in t.items()
            }
            for t in targets
        ]

        # Forward pass
        outputs = model(samples)

        # 1. Validation Loss
        loss_dict = criterion(outputs, targets)
        loss_meter.update(loss_dict["loss"].item(), samples.shape[0])

        # 2. Study Level Accuracy
        # outputs['pred_study_logits']: (B, Num_Study_Classes)
        pred_study = outputs["pred_study_logits"].argmax(dim=1)
        gt_study = torch.stack([t["study_label"] for t in targets])
        acc = (pred_study == gt_study).float().mean()
        study_acc_meter.update(acc.item(), samples.shape[0])

        # 3. Object Detection (Prepare for mAP)
        # We need to un-normalize boxes to the image size used (Config.IMG_SIZE)
        img_size = Config.IMG_SIZE

        for i in range(len(targets)):
            # --- Predictions ---
            # Probabilities: Sigmoid of class 0 (Opacity)
            # outputs['pred_logits']: (B, Q, Num_Classes+1)
            probs = outputs["pred_logits"][i, :, 0].sigmoid()

            # Boxes: Normalized (cx, cy, w, h) -> (B, Q, 4)
            boxes_norm = outputs["pred_boxes"][i]

            # Convert to absolute xyxy
            boxes_xyxy = box_cxcywh_to_xyxy(boxes_norm)
            boxes_xyxy[:, 0::2] *= img_size
            boxes_xyxy[:, 1::2] *= img_size

            # Labels: All are class 0 (Opacity)
            labels = torch.zeros_like(probs, dtype=torch.long)

            pred_boxes_list.append(boxes_xyxy.cpu())
            pred_scores_list.append(probs.cpu())
            pred_labels_list.append(labels.cpu())

            # --- Ground Truth ---
            gt_boxes_norm = targets[i]["boxes"]  # Normalized cxcywh

            if len(gt_boxes_norm) > 0:
                gt_boxes_xyxy = box_cxcywh_to_xyxy(gt_boxes_norm)
                gt_boxes_xyxy[:, 0::2] *= img_size
                gt_boxes_xyxy[:, 1::2] *= img_size
                gt_labels = targets[i]["labels"]
            else:
                gt_boxes_xyxy = torch.empty((0, 4))
                gt_labels = torch.empty((0,), dtype=torch.long)

            gt_boxes_list.append(gt_boxes_xyxy.cpu())
            gt_labels_list.append(gt_labels.cpu())

    # Calculate mAP
    map_score = calculate_map(
        pred_boxes_list,
        pred_scores_list,
        pred_labels_list,
        gt_boxes_list,
        gt_labels_list,
        num_classes=Config.NUM_OBJECT_CLASSES,
        iou_threshold=0.5,
    )

    print(f"Validation Loss: {loss_meter.avg}")
    print(f"Validation Study Acc: {study_acc_meter.avg}")
    print(f"Validation mAP: {map_score}")

    return {"loss": loss_meter.avg, "study_acc": study_acc_meter.avg, "map": map_score}

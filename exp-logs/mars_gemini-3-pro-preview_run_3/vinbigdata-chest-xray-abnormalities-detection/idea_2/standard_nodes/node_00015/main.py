import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from collections import defaultdict
import torchvision.ops as ops

# Import from provided libraries
from library.config import Config
from library.dataset import VinDrDataset
from library.model import get_model
from library.engine import fit, inference
from library.utils import collate_fn, seed_everything


def calculate_iou(boxes1, boxes2):
    """
    Calculates IoU between two sets of boxes.
    boxes1: (N, 4)
    boxes2: (M, 4)
    Returns: (N, M) matrix of IoUs
    """
    return ops.box_iou(boxes1, boxes2)


def compute_ap_voc2010(recalls, precisions):
    """
    Computes Average Precision using VOC 2010 method (interpolated precision).
    """
    # Append sentinel values at the end
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # To calculate area under PR curve, look for points where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # Sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def evaluate_map(model, data_loader, device, iou_threshold=0.4):
    """
    Evaluates mAP @ IoU > 0.4 using PASCAL VOC 2010 metric.
    """
    model.eval()

    # Store predictions and ground truths per class
    # keys: dataset_class_id (0-13)
    # values: list of entries
    preds_by_class = defaultdict(list)
    gts_by_class = defaultdict(list)

    # Track total GTs per class to calculate recall
    gt_counter = defaultdict(int)

    print("Evaluating Validation Set for mAP...")

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)

            # Get model predictions
            outputs = model(images)

            for i, output in enumerate(outputs):
                target = targets[i]

                # Ground Truth Processing
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()  # Model IDs (1-14)

                # Map Model IDs back to Dataset IDs (0-13)
                gt_dataset_ids = []
                for lbl in gt_labels:
                    if lbl in Config.MODEL_TO_DATASET_MAPPING:
                        did = Config.MODEL_TO_DATASET_MAPPING[lbl]
                        gt_dataset_ids.append(did)
                        gt_counter[did] += 1

                # Store GT info for this image
                # We need to track which GTs have been detected
                gt_info = {
                    "boxes": gt_boxes,
                    "labels": np.array(gt_dataset_ids),
                    "detected": [False] * len(gt_boxes),
                }

                # Prediction Processing
                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()
                pred_labels = output["labels"].cpu().numpy()  # Model IDs

                # Store predictions
                for idx, score in enumerate(pred_scores):
                    if score < 0.001:
                        continue  # Optimization: ignore very low confidence

                    lbl = pred_labels[idx]
                    if lbl in Config.MODEL_TO_DATASET_MAPPING:
                        did = Config.MODEL_TO_DATASET_MAPPING[lbl]
                        preds_by_class[did].append(
                            {
                                "score": score,
                                "box": pred_boxes[idx],
                                "image_idx": len(
                                    gts_by_class[did]
                                ),  # Index into the GT list we are about to append
                            }
                        )

                # We store the GT info for every class relevant to this image?
                # Actually, simpler: Store list of GT objects per image globally or per class?
                # Standard way: List of GTs per class. But we need to know which image a pred belongs to.
                # Let's append the GT info to a list for each class, but since GTs are sparse,
                # we can just store the whole image GT object in a list and reference it by index.

                # Better approach:
                # gts_by_class[class_id] -> list of {boxes: [], detected: []} corresponding to images that contain this class?
                # No, standard approach:
                # For each class:
                #   preds: list of (score, box, image_id)
                #   gts: dict mapping image_id -> {boxes: [], detected: []}

                image_id = target["image_id"]

                # Populate GT dict for this image
                # Filter boxes by class
                for cls_id in range(14):  # 0-13
                    cls_mask = np.array(gt_dataset_ids) == cls_id
                    cls_boxes = gt_boxes[cls_mask]

                    gts_by_class[cls_id].append(
                        {
                            "image_id": image_id,
                            "boxes": cls_boxes,
                            "detected": np.zeros(len(cls_boxes), dtype=bool),
                        }
                    )

                # Update preds with the index of the GT entry we just added
                # The index is len(gts_by_class[cls_id]) - 1
                # But wait, we added an entry for EVERY class for this image?
                # Yes, even if empty, to keep indexing consistent if we used simple indexing.
                # Actually, let's just store the image_id in the prediction and look it up.

                # Re-loop preds to add image_id
                current_preds_start = len(
                    preds_by_class
                )  # No, preds_by_class is a dict

    # Now compute AP per class
    aps = []

    for cls_id in range(14):  # Classes 0-13
        preds = preds_by_class[cls_id]
        # Get all GT objects for this class across all images
        # We need a map: image_id -> gt_data
        gts_map = {item["image_id"]: item for item in gts_by_class[cls_id]}

        n_pos = gt_counter[cls_id]

        if n_pos == 0:
            continue  # No GT for this class

        # Sort predictions by score descending
        preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        for i, p in enumerate(preds):
            # Find corresponding GT for this image
            # We stored "image_idx" in preds loop above?
            # Wait, in the loop above I wrote: "image_idx": len(gts_by_class[did])
            # And then I appended to gts_by_class[did].
            # So preds[i]["image_idx"] points directly to the GT object in gts_by_class[did] list.

            gt_data = gts_by_class[cls_id][p["image_idx"]]

            bb = p["box"]
            ovmax = -np.inf
            jmax = -1

            if len(gt_data["boxes"]) > 0:
                # Compute IoU
                # bb: [x1, y1, x2, y2]
                # gt_boxes: [N, 4]

                ixmin = np.maximum(gt_data["boxes"][:, 0], bb[0])
                iymin = np.maximum(gt_data["boxes"][:, 1], bb[1])
                ixmax = np.minimum(gt_data["boxes"][:, 2], bb[2])
                iymax = np.minimum(gt_data["boxes"][:, 3], bb[3])

                iw = np.maximum(ixmax - ixmin, 0.0)
                ih = np.maximum(iymax - iymin, 0.0)
                iters = iw * ih

                uni = (
                    (bb[2] - bb[0]) * (bb[3] - bb[1])
                    + (gt_data["boxes"][:, 2] - gt_data["boxes"][:, 0])
                    * (gt_data["boxes"][:, 3] - gt_data["boxes"][:, 1])
                    - iters
                )

                overlaps = iters / uni
                ovmax = np.max(overlaps)
                jmax = np.argmax(overlaps)

            if ovmax > iou_threshold:
                if not gt_data["detected"][jmax]:
                    tp[i] = 1.0
                    gt_data["detected"][jmax] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute precision recall
        fp_cumsum = np.cumsum(fp)
        tp_cumsum = np.cumsum(tp)

        recalls = tp_cumsum / n_pos
        precisions = tp_cumsum / np.maximum(
            tp_cumsum + fp_cumsum, np.finfo(np.float64).eps
        )

        ap = compute_ap_voc2010(recalls, precisions)
        aps.append(ap)
        # print(f"Class {cls_id} AP: {ap:.4f}")

    mAP = np.mean(aps) if aps else 0.0
    return mAP


def perform_failure_analysis(model, data_loader, device):
    """
    Analyzes false negatives vs number of annotations.
    """
    model.eval()

    missed_counts = []
    num_annotations = []

    print("Running Failure Analysis...")

    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            outputs = model(images)

            for i, output in enumerate(outputs):
                target = targets[i]
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                # Filter out background/no-finding from GT count logic if necessary
                # But here target boxes only exist for findings (class 0-13)
                n_gt = len(gt_boxes)

                if n_gt == 0:
                    continue

                pred_boxes = output["boxes"].cpu().numpy()
                pred_scores = output["scores"].cpu().numpy()

                # Filter preds by confidence
                keep = pred_scores > Config.CONFIDENCE_THRESHOLD
                pred_boxes = pred_boxes[keep]

                # Calculate matches
                if len(pred_boxes) == 0:
                    missed = n_gt
                else:
                    ious = calculate_iou(
                        torch.tensor(gt_boxes), torch.tensor(pred_boxes)
                    ).numpy()
                    # A GT is missed if its max IoU with any pred is < 0.4
                    if ious.shape[1] > 0:
                        max_ious = np.max(ious, axis=1)
                        missed = np.sum(max_ious < 0.4)
                    else:
                        missed = n_gt

                missed_counts.append(missed)
                num_annotations.append(n_gt)

    if len(missed_counts) > 0:
        correlation = np.corrcoef(missed_counts, num_annotations)[0, 1]
        print(
            f"Correlation between Error (Missed Count) and Input Features (Num Annotations): {correlation:.4f}"
        )
    else:
        print("Insufficient data for failure analysis correlation.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    # 2. Override Config for Fast Baseline
    # Using 50% of data and 5 epochs to fit within 1 hour limit
    # ResNet50 @ 1024x1024 is heavy.
    Config.DATASET_FRACTION = 0.5
    Config.NUM_EPOCHS = 5
    Config.BATCH_SIZE = 8  # Ensure this fits in A100 40GB

    print(f"Config: Fraction={Config.DATASET_FRACTION}, Epochs={Config.NUM_EPOCHS}")

    # 3. Data Loading
    print("Loading datasets...")
    # Note: VinDrDataset uses Config internally, so we modified Config class attributes before init
    train_dataset = VinDrDataset(
        mode="train", dataset_fraction=Config.DATASET_FRACTION, load_cached_data=True
    )
    val_dataset = VinDrDataset(
        mode="val", dataset_fraction=1.0, load_cached_data=True
    )  # Validate on full set

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

    # 4. Model Setup
    print("Initializing model...")
    model = get_model(num_classes=Config.NUM_CLASSES)
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

    # 5. Training
    print("Starting training...")
    model = fit(
        model,
        train_loader,
        val_loader,
        optimizer,
        lr_scheduler,
        device,
        num_epochs=Config.NUM_EPOCHS,
        patience=3,
    )

    # 6. Validation Metric
    mAP = evaluate_map(model, val_loader, device, iou_threshold=0.4)
    print(f"Final Validation Metric: {mAP}")

    # 7. Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 8. Submission
    THRESHOLD = 0.2646659736645272

    if mAP > THRESHOLD:
        print("Validation metric meets threshold. Generating submission...")

        test_dataset = VinDrDataset(mode="test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        df_sub = inference(model, test_loader, device)

        os.makedirs(os.path.dirname(Config.SUBMISSION_PATH), exist_ok=True)
        df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")
    else:
        print(
            f"Validation metric {mAP} did not meet threshold {THRESHOLD}. Skipping submission."
        )


if __name__ == "__main__":
    main()

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import warnings

# Import from the provided library files
from library.config import Config
from library.dataset import SIIMDataset, get_transforms
from library.model import get_model
from library.engine import Engine
from library.utils import seed_everything, collate_fn

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def calculate_iou(box1, box2):
    """
    Calculates IoU between two boxes [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection
    return intersection / union if union > 0 else 0.0


def calculate_ap(recalls, precisions):
    """
    Computes Average Precision using the VOC 2010 method (Area Under Curve).
    """
    recalls = np.concatenate(([0.0], recalls, [1.0]))
    precisions = np.concatenate(([0.0], precisions, [0.0]))

    # Compute the precision envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Integrate area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    return ap


def evaluate_map(model, data_loader, device, num_classes=4):
    """
    Evaluates mAP @ IoU > 0.5 on the validation set.
    """
    model.eval()

    # Store all GT and Predictions
    # structure: {class_id: [{'box': [], 'score': float, 'img_id': int}]}
    gt_data = {c: [] for c in range(1, num_classes)}
    pred_data = {c: [] for c in range(1, num_classes)}

    print("Running validation inference for mAP calculation...")

    with torch.no_grad():
        for images, targets, image_ids in data_loader:
            images = list(img.to(device) for img in images)

            # Get Predictions
            outputs = model(images)
            outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]

            # Process Batch
            for i, output in enumerate(outputs):
                img_id = image_ids[i]
                target = targets[i]

                # Ground Truth
                gt_boxes = target["boxes"].numpy()
                gt_labels = target["labels"].numpy()

                for box, label in zip(gt_boxes, gt_labels):
                    if label in gt_data:
                        gt_data[label].append(
                            {"box": box, "used": False, "img_id": img_id}
                        )

                # Predictions
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]

                for box, score, label in zip(pred_boxes, pred_scores, pred_labels):
                    if label in pred_data:
                        pred_data[label].append(
                            {"box": box, "score": score, "img_id": img_id}
                        )

    # Calculate AP per class
    aps = []

    for c in range(1, num_classes):
        gts = gt_data[c]
        preds = pred_data[c]

        # Sort predictions by confidence descending
        preds.sort(key=lambda x: x["score"], reverse=True)

        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))

        # Total ground truths for this class
        n_pos = len(gts)

        if n_pos == 0:
            # If no ground truth for this class, AP is 0 unless no preds either (undefined, usually 0)
            aps.append(0.0)
            continue

        # Group GT by image for faster lookup
        gts_by_image = {}
        for idx, gt in enumerate(gts):
            iid = gt["img_id"]
            if iid not in gts_by_image:
                gts_by_image[iid] = []
            gts_by_image[iid].append(idx)  # Store index in original gts list

        for i, pred in enumerate(preds):
            img_id = pred["img_id"]
            pred_box = pred["box"]

            best_iou = 0.0
            best_gt_idx = -1

            if img_id in gts_by_image:
                # Check against all GTs in this image
                for gt_idx in gts_by_image[img_id]:
                    iou = calculate_iou(pred_box, gts[gt_idx]["box"])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = gt_idx

            if best_iou > 0.5:
                if not gts[best_gt_idx]["used"]:
                    tp[i] = 1.0
                    gts[best_gt_idx]["used"] = True
                else:
                    fp[i] = 1.0
            else:
                fp[i] = 1.0

        # Compute Precision and Recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)

        recalls = tp_cumsum / n_pos
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)

        ap = calculate_ap(recalls, precisions)
        aps.append(ap)

    return np.mean(aps)


def perform_failure_analysis(model, data_loader, device):
    """
    Calculates loss per image and correlates it with metadata features.
    """
    print("\nPerforming Failure Analysis...")

    # To get loss, model must be in train mode
    model.train()

    results = []

    with torch.no_grad():
        for images, targets, image_ids in data_loader:
            images = list(img.to(device) for img in images)
            targets_gpu = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Forward pass to get losses
            # Note: torchvision returns a dict of losses. We sum them.
            # However, standard forward() averages over batch.
            # We need per-image loss. We process one by one effectively or assume batch average approximates.
            # For accurate per-image analysis, we pass images one by one or extract unreduced loss if possible.
            # The standard model reduces loss. We will iterate batch items manually to get per-item loss.

            for i in range(len(images)):
                single_img = [images[i]]
                single_target = [targets_gpu[i]]

                loss_dict = model(single_img, single_target)
                total_loss = sum(loss for loss in loss_dict.values()).item()

                # Extract features from CPU target
                tgt = targets[i]
                num_boxes = len(tgt["boxes"])

                if num_boxes > 0:
                    areas = (tgt["boxes"][:, 2] - tgt["boxes"][:, 0]) * (
                        tgt["boxes"][:, 3] - tgt["boxes"][:, 1]
                    )
                    avg_area = areas.mean().item()
                else:
                    avg_area = 0.0

                results.append(
                    {"loss": total_loss, "num_boxes": num_boxes, "avg_area": avg_area}
                )

    df = pd.DataFrame(results)

    # Calculate correlations
    if len(df) > 0:
        corr_boxes = df["loss"].corr(df["num_boxes"])
        corr_area = df["loss"].corr(df["avg_area"])

        print("Correlation between Error (Loss) and Input Features:")
        print(f"  Loss vs Num Boxes: {corr_boxes:.4f}")
        print(f"  Loss vs Avg Box Area: {corr_area:.4f}")
    else:
        print("No data for failure analysis.")


def main():
    # 1. Setup
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 2. Data Loading
    # Load metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Use full dataset for valid metric calculation
    train_dataset = SIIMDataset(
        train_df,
        mode="train",
        transforms=get_transforms("train"),
        limit_size=None,
    )

    val_dataset = SIIMDataset(
        val_df, mode="val", transforms=get_transforms("val"), limit_size=None
    )

    test_dataset = SIIMDataset(test_df, mode="test", transforms=get_transforms("test"))

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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    print(
        f"Data loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    # 3. Model Initialization
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

    # 4. Training
    engine = Engine(model, device, optimizer, lr_scheduler)

    # Use configured epochs for full training
    EPOCHS = Config.EPOCHS
    # Increase patience to 5 to allow for convergence with stronger augmentation
    best_model_path = engine.fit_model(
        train_loader, val_loader, epochs=EPOCHS, patience=5
    )

    # 5. Validation Assessment
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    # Calculate mAP
    map_score = evaluate_map(model, val_loader, device, num_classes=Config.NUM_CLASSES)
    print(f"Final Validation Metric: {map_score}")

    # Failure Analysis
    perform_failure_analysis(model, val_loader, device)

    # 6. Inference and Submission
    # Only generate submission if metric exceeds the target threshold
    TARGET_METRIC = 0.19051633522746228
    if map_score > TARGET_METRIC:
        engine.inference(test_loader, best_model_path)
    else:
        print(
            f"Validation metric {map_score} did not beat target {TARGET_METRIC}. Skipping submission."
        )


if __name__ == "__main__":
    main()

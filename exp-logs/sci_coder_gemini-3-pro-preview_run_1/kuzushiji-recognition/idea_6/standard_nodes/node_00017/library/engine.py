import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import (
    WORKING_DIR,
    SUBMISSION_PATH,
    TRAIN_METADATA_PATH,
    VAL_METADATA_PATH,
    TEST_METADATA_PATH,
    BATCH_SIZE,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_WORKERS,
    DEVICE,
    IMG_SIZE,
    DOWN_RATIO,
    CONF_THRESHOLD,
    MAX_DETECTIONS,
    BACKBONE,
)
from library.utils import (
    seed_everything,
    Logger,
    collate_fn,
    post_process_coords,
    load_and_parse_metadata,
)
from library.dataset import KuzushijiDataset, get_class_mapping
from library.model import ConvNextCenterNet
from library.loss import CenterNetLoss, _transpose_and_gather_feat


def _nms(heatmap, kernel=3):
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _decode(hm, wh, reg, cls_logits, k=1200):
    """
    Decodes the model output into bounding box centers and class indices.
    """
    batch_size, _, height, width = hm.shape

    # Apply NMS to heatmap
    hm = torch.sigmoid(hm)
    hm = _nms(hm)

    # Flatten
    hm_flat = hm.view(batch_size, -1)

    # Select top K peaks
    scores, inds = torch.topk(hm_flat, k)

    # Convert indices to grid coordinates
    ys = inds.div(width, rounding_mode="floor")
    xs = inds % width

    # Gather regression offsets
    # reg: (B, 2, H, W) -> (B, K, 2)
    reg = _transpose_and_gather_feat(reg, inds)
    reg = reg.view(batch_size, k, 2)

    # Apply offsets to grid coordinates
    xs = xs.view(batch_size, k, 1) + reg[:, :, 0:1]
    ys = ys.view(batch_size, k, 1) + reg[:, :, 1:2]

    # Gather class logits
    # cls_logits: (B, Num_Classes, H, W) -> (B, K, Num_Classes)
    cls_logits = _transpose_and_gather_feat(cls_logits, inds)

    # Get most likely class for each point
    cls_scores, cls_ids = torch.max(cls_logits, dim=2)

    # Final scores could be a combination, but we use objectness (hm)
    # masked by class confidence if needed. Here we stick to hm score.

    return scores, xs, ys, cls_ids


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for batch in dataloader:
        images = batch["image"].to(device)

        # Move targets to device
        targets = batch["target"]
        for k, v in targets.items():
            if isinstance(v, torch.Tensor):
                targets[k] = v.to(device)

        batch["target"] = targets

        optimizer.zero_grad()

        outputs = model(images)
        loss, _ = criterion(outputs, batch)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def evaluate(model, dataloader, criterion, device, metadata_path):
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    # Load GT for F1 calculation
    # We load metadata to get the raw ground truth boxes for metric calculation
    gt_data = load_and_parse_metadata(metadata_path)
    gt_map = {item["image_id"]: item["annotations"] for item in gt_data}

    # Get class mapping
    char_to_id, id_to_char = get_class_mapping()

    tp_global = 0
    fp_global = 0
    fn_global = 0

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = batch["target"]
            image_ids = batch["image_id"]
            original_shapes = batch["original_shape"]

            for k, v in targets.items():
                if isinstance(v, torch.Tensor):
                    targets[k] = v.to(device)
            batch["target"] = targets

            # Forward
            outputs = model(images)
            loss, _ = criterion(outputs, batch)

            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Decode for Metric
            scores, xs, ys, cls_ids = _decode(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                outputs["cls_logits"],
                k=MAX_DETECTIONS,
            )

            # Process each image in batch
            for i in range(batch_size):
                img_id = image_ids[i]
                orig_shape = original_shapes[i]

                # Get Predictions for this image
                # Filter by confidence
                mask = scores[i] > CONF_THRESHOLD
                if mask.sum() == 0:
                    pred_points = []
                    pred_classes = []
                    pred_scores = []
                else:
                    valid_scores = scores[i][mask]
                    valid_xs = xs[i][mask]
                    valid_ys = ys[i][mask]
                    valid_cls = cls_ids[i][mask]

                    # Convert to numpy
                    v_xs = valid_xs.cpu().numpy().flatten()
                    v_ys = valid_ys.cpu().numpy().flatten()
                    v_cls = valid_cls.cpu().numpy().flatten()
                    v_scores = valid_scores.cpu().numpy().flatten()

                    pred_points = []
                    pred_classes = []
                    pred_scores = []

                    for j in range(len(v_scores)):
                        # Scale up to input size then map to original
                        # Model output is DOWN_RATIO smaller than input (1024)
                        # xs, ys are in feature map coordinates

                        # Map to input image coords (1024x1024)
                        ix = v_xs[j] * DOWN_RATIO
                        iy = v_ys[j] * DOWN_RATIO

                        # Map to original image coords
                        ox, oy = post_process_coords(ix, iy, orig_shape, IMG_SIZE)

                        pred_points.append((ox, oy))
                        pred_classes.append(id_to_char[v_cls[j]])
                        pred_scores.append(v_scores[j])

                # Get GT for this image
                gt_anns = gt_map.get(img_id, [])

                # Matching Logic
                # Sort predictions by score
                preds = sorted(
                    zip(pred_points, pred_classes, pred_scores),
                    key=lambda x: x[2],
                    reverse=True,
                )

                # Track matched GT indices
                matched_gt = set()

                curr_tp = 0
                curr_fp = 0

                for (px, py), p_label, p_score in preds:
                    match_found = False
                    # Check against all unmatched GTs
                    for gt_idx, ann in enumerate(gt_anns):
                        if gt_idx in matched_gt:
                            continue

                        if ann["label"] != p_label:
                            continue

                        gx, gy, gw, gh = ann["bbox"]

                        # Check if point inside box
                        if (gx <= px <= gx + gw) and (gy <= py <= gy + gh):
                            matched_gt.add(gt_idx)
                            match_found = True
                            break

                    if match_found:
                        curr_tp += 1
                    else:
                        curr_fp += 1

                curr_fn = len(gt_anns) - len(matched_gt)

                tp_global += curr_tp
                fp_global += curr_fp
                fn_global += curr_fn

    avg_loss = running_loss / dataset_size

    # Calculate F1
    precision = tp_global / (tp_global + fp_global + 1e-8)
    recall = tp_global / (tp_global + fn_global + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)

    return avg_loss, f1


def run_training(debug=False, epochs=NUM_EPOCHS):
    seed_everything()

    # Datasets
    train_dataset = KuzushijiDataset(TRAIN_METADATA_PATH, split="train", debug=debug)
    val_dataset = KuzushijiDataset(VAL_METADATA_PATH, split="val", debug=debug)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Model Setup
    model = ConvNextCenterNet().to(DEVICE)
    criterion = CenterNetLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    logger = Logger(os.path.join(WORKING_DIR, "training_log.csv"))

    best_f1 = 0.0
    best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs on {DEVICE}...")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, DEVICE, epoch
        )
        val_loss, val_f1 = evaluate(
            model, val_loader, criterion, DEVICE, VAL_METADATA_PATH
        )

        scheduler.step()

        logger.log(epoch, train_loss, val_loss, val_f1, "N/A")

        # Checkpoint
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with F1: {best_f1}")

    print(f"Training complete. Best F1: {best_f1}")
    return best_model_path


def predict(model_path=None, debug=False):
    seed_everything()

    if model_path is None:
        model_path = os.path.join(WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("No model found for inference.")
        return

    # Load Model
    model = ConvNextCenterNet().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    # Dataset
    test_dataset = KuzushijiDataset(TEST_METADATA_PATH, split="test", debug=debug)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    _, id_to_char = get_class_mapping()

    results = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(DEVICE)
            image_ids = batch["image_id"]
            original_shapes = batch["original_shape"]

            outputs = model(images)

            scores, xs, ys, cls_ids = _decode(
                outputs["hm"],
                outputs["wh"],
                outputs["reg"],
                outputs["cls_logits"],
                k=MAX_DETECTIONS,
            )

            batch_size = images.size(0)

            for i in range(batch_size):
                img_id = image_ids[i]
                orig_shape = original_shapes[i]

                mask = scores[i] > CONF_THRESHOLD

                label_strings = []

                if mask.sum() > 0:
                    valid_xs = xs[i][mask].cpu().numpy().flatten()
                    valid_ys = ys[i][mask].cpu().numpy().flatten()
                    valid_cls = cls_ids[i][mask].cpu().numpy().flatten()

                    for j in range(len(valid_xs)):
                        ix = valid_xs[j] * DOWN_RATIO
                        iy = valid_ys[j] * DOWN_RATIO

                        ox, oy = post_process_coords(ix, iy, orig_shape, IMG_SIZE)

                        char_label = id_to_char[valid_cls[j]]

                        # Format: Unicode X Y
                        label_strings.append(f"{char_label} {int(ox)} {int(oy)}")

                # Join all labels for this image
                full_label_str = " ".join(label_strings)
                results.append({"image_id": img_id, "labels": full_label_str})

    # Save Submission
    df = pd.DataFrame(results)
    df.to_csv(SUBMISSION_PATH, index=False)
    print(f"Submission saved to {SUBMISSION_PATH}")

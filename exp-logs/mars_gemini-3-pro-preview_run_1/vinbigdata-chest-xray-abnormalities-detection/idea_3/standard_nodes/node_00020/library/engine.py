import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from torchvision.ops import box_iou
import time
from torch.amp import autocast, GradScaler

from library.config import Config
from library.loss import CenterNetLoss
from library.utils import save_checkpoint, get_original_dimensions


def decode_detections(hm, wh, reg, k=100):
    """
    Decodes the output of the CenterNet model into bounding boxes.

    Args:
        hm: Heatmap (B, C, H, W)
        wh: Width/Height (B, 2, H, W)
        reg: Offset (B, 2, H, W)
        k: Top K detections to keep

    Returns:
        dets: (B, K, 6) [x1, y1, x2, y2, score, class]
    """
    batch_size, num_classes, height, width = hm.shape

    # 1. Max Pooling to find peaks (NMS)
    hm_max = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hm_max == hm).float()
    hm = hm * keep

    # 2. Top K
    # Flatten to (B, C*H*W)
    hm_flat = hm.view(batch_size, -1)
    topk_scores, topk_inds = torch.topk(hm_flat, k)

    topk_clses = (topk_inds // (height * width)).float()
    topk_inds = topk_inds % (height * width)

    topk_ys = (topk_inds // width).float()
    topk_xs = (topk_inds % width).float()

    # 3. Retrieve Reg and WH at peak locations
    # reg: (B, 2, H, W) -> (B, 2, H*W)
    reg = reg.view(batch_size, 2, -1)
    wh = wh.view(batch_size, 2, -1)

    # Gather values
    # We need to expand indices to match dimensions for gather
    inds_expand = topk_inds.unsqueeze(1)  # (B, 1, K)

    reg_x = torch.gather(reg[:, 0, :], 1, topk_inds)
    reg_y = torch.gather(reg[:, 1, :], 1, topk_inds)

    w_det = torch.gather(wh[:, 0, :], 1, topk_inds)
    h_det = torch.gather(wh[:, 1, :], 1, topk_inds)

    # 4. Apply Offsets and Stride
    # The model output is stride 4 (160x160 for 640 input)
    # We work in feature map coords first, then scale

    xs = topk_xs + reg_x
    ys = topk_ys + reg_y

    x1 = xs - w_det / 2
    y1 = ys - h_det / 2
    x2 = xs + w_det / 2
    y2 = ys + h_det / 2

    # Scale to input resolution (stride 4)
    x1 = x1 * 4
    y1 = y1 * 4
    x2 = x2 * 4
    y2 = y2 * 4

    # Stack: (B, K, 6)
    # [x1, y1, x2, y2, score, class]
    dets = torch.stack([x1, y1, x2, y2, topk_scores, topk_clses], dim=2)

    return dets


def calculate_map(pred_df, gt_df, iou_thresh=0.4):
    """
    Calculates PASCAL VOC mAP @ IoU > 0.4 for classes 0-13.
    """
    average_precisions = []

    # Classes 0 to 13
    target_classes = [c for c in range(14)]

    for c in target_classes:
        # Filter data for this class
        preds = pred_df[pred_df["class_id"] == c].copy()
        gts = gt_df[gt_df["class_id"] == c].copy()

        n_pos = len(gts)
        if n_pos == 0:
            # If no ground truth for this class, AP is 0 unless no predictions (then undefined/skipped)
            # Usually we count it as 0 if there were predictions, or skip.
            # Standard practice: if no GT, AP is 0.
            average_precisions.append(0.0)
            continue

        # Sort predictions by confidence
        preds = preds.sort_values("confidence", ascending=False).reset_index(drop=True)

        TP = np.zeros(len(preds))
        FP = np.zeros(len(preds))

        # Track which GT boxes have been matched
        gt_matched = {
            img_id: np.zeros(len(gts[gts["image_id"] == img_id]))
            for img_id in gts["image_id"].unique()
        }

        for i, row in preds.iterrows():
            img_id = row["image_id"]
            pred_box = torch.tensor(
                [[row["x_min"], row["y_min"], row["x_max"], row["y_max"]]]
            )

            # Get GT for this image and class
            img_gts = gts[gts["image_id"] == img_id]

            if len(img_gts) == 0:
                FP[i] = 1
                continue

            gt_boxes = torch.tensor(
                img_gts[["x_min", "y_min", "x_max", "y_max"]].values
            )

            # Calculate IoU
            ious = box_iou(pred_box, gt_boxes)[0]

            if len(ious) > 0:
                max_iou, max_idx = torch.max(ious, dim=0)
                max_iou = max_iou.item()
                max_idx = max_idx.item()

                if max_iou > iou_thresh:
                    if gt_matched[img_id][max_idx] == 0:
                        TP[i] = 1
                        gt_matched[img_id][max_idx] = 1
                    else:
                        FP[i] = 1  # Duplicate detection
                else:
                    FP[i] = 1
            else:
                FP[i] = 1

        # Compute Precision and Recall
        acc_TP = np.cumsum(TP)
        acc_FP = np.cumsum(FP)

        rec = acc_TP / n_pos
        prec = acc_TP / (acc_TP + acc_FP + 1e-6)

        # Compute AP (VOC 2010 uses all points / integration)
        # We use the standard 11-point or area under curve.
        # Using simple area under curve approximation here
        ap = 0.0
        # Add sentinel values
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))

        # Compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # Integrate area under curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        average_precisions.append(ap)

    return np.mean(average_precisions)


def train_one_epoch(
    model, dataloader, optimizer, device, epoch, scheduler=None, scaler=None
):
    model.train()
    loss_meter = {
        "loss": 0.0,
        "hm_loss": 0.0,
        "wh_loss": 0.0,
        "off_loss": 0.0,
        "global_loss": 0.0,
    }
    criterion = CenterNetLoss()

    # No progress bar as per requirements, just iterate
    for batch_idx, (images, targets, _) in enumerate(dataloader):
        images = images.to(device)

        # Mixed Precision Forward
        with autocast("cuda"):
            outputs = model(images)
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["loss"]

        # Backward with Scaler
        optimizer.zero_grad()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # Update metrics
        for k, v in loss_dict.items():
            loss_meter[k] += v.item()

    # Average losses
    n = len(dataloader)
    for k in loss_meter:
        loss_meter[k] /= n

    print(
        f"Epoch {epoch} Train Loss: {loss_meter['loss']:.6f} "
        f"(HM: {loss_meter['hm_loss']:.4f}, WH: {loss_meter['wh_loss']:.4f}, "
        f"Off: {loss_meter['off_loss']:.4f}, Global: {loss_meter['global_loss']:.4f})"
    )

    return loss_meter["loss"]


def evaluate(model, dataloader, device, val_meta_path=Config.VAL_META_PATH):
    model.eval()

    # Load Ground Truth
    df_val = pd.read_csv(val_meta_path)
    # Filter out "No finding" from GT for mAP calculation (mAP is for classes 0-13)
    gt_df = df_val[df_val["class_id"] != Config.NO_FINDING_CLASS_ID].copy()

    # Get original dimensions for rescaling
    orig_dims = get_original_dimensions(df_val)

    predictions = []

    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = images.to(device)
            # Use autocast for inference as well to save memory
            with autocast("cuda"):
                outputs = model(images)

            # 1. Global Gating
            global_probs = (
                torch.sigmoid(outputs["global_no_finding"]).squeeze(1).cpu().numpy()
            )  # (B,)

            # 2. Decode Detections
            # dets: (B, K, 6)
            dets = decode_detections(outputs["hm"], outputs["wh"], outputs["reg"])
            dets = dets.cpu().numpy()

            for i, img_id in enumerate(image_ids):
                # Check Global Head
                if global_probs[i] > Config.GLOBAL_CLS_THRESHOLD:
                    # Predict No Finding (implicitly means no boxes for classes 0-13)
                    continue

                # Get detections for this image
                img_dets = dets[i]

                # Get original size
                orig_w, orig_h = orig_dims.get(
                    img_id, (Config.IMG_SIZE, Config.IMG_SIZE)
                )

                # Rescale factor
                scale_x = orig_w / Config.IMG_SIZE
                scale_y = orig_h / Config.IMG_SIZE

                for box in img_dets:
                    x1, y1, x2, y2, score, cls_id = box

                    if score < Config.CONF_THRESHOLD:
                        continue

                    # Rescale to original
                    x1 = np.clip(x1 * scale_x, 0, orig_w)
                    y1 = np.clip(y1 * scale_y, 0, orig_h)
                    x2 = np.clip(x2 * scale_x, 0, orig_w)
                    y2 = np.clip(y2 * scale_y, 0, orig_h)

                    # Store
                    predictions.append(
                        {
                            "image_id": img_id,
                            "class_id": int(cls_id),
                            "confidence": float(score),
                            "x_min": x1,
                            "y_min": y1,
                            "x_max": x2,
                            "y_max": y2,
                        }
                    )

    # Calculate mAP
    if not predictions:
        print("Validation: No predictions made.")
        return 0.0

    pred_df = pd.DataFrame(predictions)

    # Calculate mAP
    map_score = calculate_map(pred_df, gt_df, iou_thresh=Config.IOU_THRESHOLD)

    print(f"Validation mAP@0.4: {map_score:.10f}")
    return map_score


def train_model(model, dataloaders, device, epochs=Config.EPOCHS):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )
    # Initialize Scaler for Mixed Precision
    scaler = GradScaler("cuda")

    best_score = 0.0
    patience = 5
    patience_counter = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model,
            dataloaders["train"],
            optimizer,
            device,
            epoch,
            scaler=scaler,  # Pass scaler
        )

        # Evaluate
        if "val" in dataloaders:
            val_score = evaluate(model, dataloaders["val"], device)

            # Scheduler Step
            scheduler.step()

            # Checkpoint & Early Stopping
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                print(f"New best mAP: {best_score:.10f}. Saving model...")
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    best_score,
                    Config.MODEL_SAVE_PATH,
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break
        else:
            # If no validation set, just save last
            save_checkpoint(
                model, optimizer, scheduler, epoch, 0.0, Config.MODEL_SAVE_PATH
            )
            scheduler.step()


def predict_and_submit(model, dataloader, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    model.eval()

    # Load Test Metadata for Original Dimensions
    df_test = pd.read_csv(Config.TEST_META_PATH)
    orig_dims = get_original_dimensions(df_test)

    results = []

    print("Generating predictions for test set...")

    with torch.no_grad():
        for images, targets, image_ids in dataloader:
            images = images.to(device)
            # Use autocast for inference
            with autocast("cuda"):
                outputs = model(images)

            # Global Gating
            global_probs = (
                torch.sigmoid(outputs["global_no_finding"]).squeeze(1).cpu().numpy()
            )

            # Decode
            dets = decode_detections(outputs["hm"], outputs["wh"], outputs["reg"])
            dets = dets.cpu().numpy()

            for i, img_id in enumerate(image_ids):
                prediction_strings = []

                # Check Global "No Finding"
                is_no_finding = global_probs[i] > Config.GLOBAL_CLS_THRESHOLD

                if not is_no_finding:
                    # Process detections
                    img_dets = dets[i]
                    orig_w, orig_h = orig_dims.get(
                        img_id, (Config.IMG_SIZE, Config.IMG_SIZE)
                    )
                    scale_x = orig_w / Config.IMG_SIZE
                    scale_y = orig_h / Config.IMG_SIZE

                    found_valid_box = False

                    for box in img_dets:
                        x1, y1, x2, y2, score, cls_id = box

                        if score > Config.CONF_THRESHOLD:
                            # Rescale
                            x1 = np.clip(x1 * scale_x, 0, orig_w)
                            y1 = np.clip(y1 * scale_y, 0, orig_h)
                            x2 = np.clip(x2 * scale_x, 0, orig_w)
                            y2 = np.clip(y2 * scale_y, 0, orig_h)

                            # Format: class_id confidence xmin ymin xmax ymax
                            pred_str = f"{int(cls_id)} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                            prediction_strings.append(pred_str)
                            found_valid_box = True

                    if not found_valid_box:
                        # If detections were filtered out by threshold, default to No Finding
                        is_no_finding = True

                if is_no_finding:
                    # "14 1 0 0 1 1"
                    prediction_strings.append(
                        f"{Config.NO_FINDING_CLASS_ID} 1.0 0 0 1 1"
                    )

                results.append(
                    {
                        "image_id": img_id,
                        "PredictionString": " ".join(prediction_strings),
                    }
                )

    submission_df = pd.DataFrame(results)
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

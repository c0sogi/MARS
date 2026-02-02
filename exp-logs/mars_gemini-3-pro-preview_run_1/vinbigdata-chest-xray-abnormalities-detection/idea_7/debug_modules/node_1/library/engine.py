import os
import torch
import numpy as np
import pandas as pd
import torchvision
from library.config import Config
from library.utils import calculate_map
from library.loss import CenterNetLoss


def decode_detections(outputs, original_dims, image_ids):
    """
    Decodes model outputs into bounding boxes in original image coordinates.
    """
    heatmap = torch.sigmoid(outputs["heatmap"])
    wh = outputs["wh"]
    offset = outputs["offset"]
    global_probs = torch.sigmoid(outputs["global_logits"])

    B, C, H, W = heatmap.shape

    # 1. Find Peaks using Max Pooling (3x3)
    hmax = torch.nn.functional.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    # Flatten for top-k extraction
    heatmap_flat = heatmap.view(B, -1)

    results = []

    for i in range(B):
        img_id = image_ids[i]
        orig_h, orig_w = original_dims[i]
        g_prob = global_probs[i].item()

        # --- Global Gate ---
        # If "No Finding" probability is high, output the specific no-finding prediction
        if g_prob > Config.GLOBAL_THRESHOLD:
            results.append(
                {
                    "image_id": img_id,
                    "class_id": Config.CLASS_ID_NO_FINDING,
                    "confidence": 1.0,
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 1,
                    "y_max": 1,
                }
            )
            continue

        # --- Extract Detections ---
        # Get top detections
        top_k = 100
        scores, indices = torch.topk(
            heatmap_flat[i], k=min(top_k, heatmap_flat.size(1))
        )

        # Filter by confidence threshold
        mask = scores > Config.CONF_THRESHOLD
        scores = scores[mask]
        indices = indices[mask]

        if len(scores) == 0:
            # If no boxes found but global gate didn't trigger,
            # we technically have "No finding"
            results.append(
                {
                    "image_id": img_id,
                    "class_id": Config.CLASS_ID_NO_FINDING,
                    "confidence": 1.0,
                    "x_min": 0,
                    "y_min": 0,
                    "x_max": 1,
                    "y_max": 1,
                }
            )
            continue

        # Convert indices to coordinates
        # indices are flat indices in (C * H * W)
        area = H * W
        class_ids = indices // area
        spatial_indices = indices % area
        ys = spatial_indices // W
        xs = spatial_indices % W

        # Gather regression targets
        # wh and offset are (2, H, W)
        # We flatten spatial dims to index easily
        wh_flat = wh[i].view(2, -1)
        off_flat = offset[i].view(2, -1)

        w_pred = wh_flat[0, spatial_indices]
        h_pred = wh_flat[1, spatial_indices]
        ox_pred = off_flat[0, spatial_indices]
        oy_pred = off_flat[1, spatial_indices]

        # Reconstruct Box (Stride 4)
        cx = xs.float() + ox_pred
        cy = ys.float() + oy_pred

        x1 = (cx - w_pred / 2) * 4
        y1 = (cy - h_pred / 2) * 4
        x2 = (cx + w_pred / 2) * 4
        y2 = (cy + h_pred / 2) * 4

        # --- Rescale to Original Dimensions ---
        scale_x = orig_w.float() / Config.IMG_SIZE
        scale_y = orig_h.float() / Config.IMG_SIZE

        x1 = x1 * scale_x
        x2 = x2 * scale_x
        y1 = y1 * scale_y
        y2 = y2 * scale_y

        # Clip to image boundaries
        x1 = torch.clamp(x1, 0, orig_w.float())
        x2 = torch.clamp(x2, 0, orig_w.float())
        y1 = torch.clamp(y1, 0, orig_h.float())
        y2 = torch.clamp(y2, 0, orig_h.float())

        # Stack for NMS
        boxes = torch.stack([x1, y1, x2, y2], dim=1)

        # --- NMS ---
        # We apply NMS per image (across all classes or per class? Usually per class,
        # but here we can do global if we assume non-overlapping distinct pathologies,
        # but standard is per-class. torchvision nms is class-agnostic on the box set provided.
        # To do per-class NMS, we offset boxes by class_id * max_coord.

        # Class-aware NMS trick
        max_coordinate = max(orig_h, orig_w).float()
        offsets = class_ids.float() * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]

        keep_indices = torchvision.ops.nms(boxes_for_nms, scores, Config.IOU_THRESHOLD)

        for k in keep_indices:
            results.append(
                {
                    "image_id": img_id,
                    "class_id": class_ids[k].item(),
                    "confidence": scores[k].item(),
                    "x_min": boxes[k, 0].item(),
                    "y_min": boxes[k, 1].item(),
                    "x_max": boxes[k, 2].item(),
                    "y_max": boxes[k, 3].item(),
                }
            )

    return results


def train_one_epoch(model, dataloader, optimizer, device, epoch):
    model.train()
    criterion = CenterNetLoss()

    loss_meter = {"total": [], "hm": [], "wh": [], "off": [], "global": []}

    for batch in dataloader:
        images = batch["image"].to(device)
        targets = {k: v.to(device) for k, v in batch["target"].items()}

        optimizer.zero_grad()
        outputs = model(images)

        loss, loss_stats = criterion(outputs, targets)

        loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

        optimizer.step()

        # Log
        loss_meter["total"].append(loss_stats["loss"].item())
        loss_meter["hm"].append(loss_stats["hm_loss"].item())
        loss_meter["wh"].append(loss_stats["wh_loss"].item())
        loss_meter["off"].append(loss_stats["off_loss"].item())
        loss_meter["global"].append(loss_stats["global_loss"].item())

    metrics = {k: np.mean(v) for k, v in loss_meter.items()}
    print(
        f"Epoch {epoch} Train Loss: {metrics['total']} (HM: {metrics['hm']}, WH: {metrics['wh']}, Off: {metrics['off']}, Global: {metrics['global']})"
    )

    return metrics["total"]


def validate(model, dataloader, device, gt_df):
    model.eval()
    criterion = CenterNetLoss()

    loss_meter = {"total": [], "hm": [], "wh": [], "off": [], "global": []}
    predictions = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            targets = {k: v.to(device) for k, v in batch["target"].items()}

            outputs = model(images)
            loss, loss_stats = criterion(outputs, targets)

            loss_meter["total"].append(loss_stats["loss"].item())
            loss_meter["hm"].append(loss_stats["hm_loss"].item())
            loss_meter["wh"].append(loss_stats["wh_loss"].item())
            loss_meter["off"].append(loss_stats["off_loss"].item())
            loss_meter["global"].append(loss_stats["global_loss"].item())

            # Decode predictions for mAP
            batch_preds = decode_detections(
                outputs, batch["original_dim"], batch["image_id"]
            )
            predictions.extend(batch_preds)

    avg_loss = np.mean(loss_meter["total"])

    # Calculate mAP
    if len(predictions) > 0:
        pred_df = pd.DataFrame(predictions)
    else:
        pred_df = pd.DataFrame(
            columns=[
                "image_id",
                "class_id",
                "confidence",
                "x_min",
                "y_min",
                "x_max",
                "y_max",
            ]
        )

    map_score = calculate_map(pred_df, gt_df, iou_threshold=Config.IOU_THRESHOLD)

    print(f"Validation Loss: {avg_loss}")
    print(f"Validation mAP: {map_score}")

    return avg_loss, map_score


def generate_submission(model, dataloader, device, output_path):
    model.eval()
    predictions = []

    print("Generating submission...")
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            # No targets in test mode

            outputs = model(images)
            batch_preds = decode_detections(
                outputs, batch["original_dim"], batch["image_id"]
            )
            predictions.extend(batch_preds)

    if not predictions:
        print("Warning: No predictions generated. Creating empty submission.")
        # Fallback to sample submission logic if needed, but decode_detections handles defaults

    # Format for submission
    # ID, PredictionString
    # String: class conf xmin ymin xmax ymax ...

    pred_df = pd.DataFrame(predictions)

    # Group by image_id
    submission_rows = []

    # Ensure all test images are present
    all_test_ids = dataloader.dataset.image_ids

    if not pred_df.empty:
        grouped = pred_df.groupby("image_id")
        for img_id in all_test_ids:
            if img_id in grouped.groups:
                group = grouped.get_group(img_id)
                strings = []
                for _, row in group.iterrows():
                    s = f"{int(row['class_id'])} {row['confidence']:.4f} {row['x_min']} {row['y_min']} {row['x_max']} {row['y_max']}"
                    strings.append(s)
                pred_str = " ".join(strings)
            else:
                # Default to No Finding
                pred_str = "14 1 0 0 1 1"

            submission_rows.append({"image_id": img_id, "PredictionString": pred_str})
    else:
        # All defaults
        for img_id in all_test_ids:
            submission_rows.append(
                {"image_id": img_id, "PredictionString": "14 1 0 0 1 1"}
            )

    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


def fit(
    model, train_loader, val_loader, optimizer, scheduler, device, num_epochs, gt_df
):
    best_map = 0.0
    patience = 5
    patience_counter = 0

    for epoch in range(1, num_epochs + 1):
        print(f"\n--- Epoch {epoch}/{num_epochs} ---")

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch)

        # Validate
        val_loss, val_map = validate(model, val_loader, device, gt_df)

        # Scheduler Step
        if scheduler:
            scheduler.step()

        # Checkpointing
        # Save Last
        torch.save(
            model.state_dict(), os.path.join(Config.CHECKPOINT_DIR, "last_model.pth")
        )

        # Save Best
        if val_map > best_map:
            best_map = val_map
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.CHECKPOINT_DIR, "best_model.pth"),
            )
            print(f"New Best mAP: {best_map} (Saved model)")
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print(
                f"Early stopping triggered after {patience} epochs of no improvement."
            )
            break

    print(f"Training completed. Best mAP: {best_map}")

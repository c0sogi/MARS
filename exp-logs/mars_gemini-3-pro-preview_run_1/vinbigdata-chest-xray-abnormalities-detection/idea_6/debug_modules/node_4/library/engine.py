import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from library.config import Config
from library.loss import ThoracicLoss
from library.utils import get_image_and_dimensions


def train_one_epoch(model, dataloader, optimizer, scheduler, device, epoch):
    """
    Trains the model for one epoch.
    """
    model.train()
    criterion = ThoracicLoss()

    running_loss = 0.0
    dataset_size = 0

    # Progress bar for feedback
    pbar = tqdm(
        dataloader, desc=f"Epoch {epoch+1}/{Config.NUM_EPOCHS} [Train]", leave=False
    )

    for batch_idx, (images, targets, image_ids) in enumerate(pbar):
        batch_size = images.size(0)

        # Move data to device
        images = images.to(device)

        # Move targets to device (dictionary of tensors)
        targets = {k: v.to(device) for k, v in targets.items()}

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss, loss_stats = criterion(outputs, targets)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.CLIP_GRAD_NORM)

        # Optimizer step
        optimizer.step()

        # Update stats
        running_loss += loss.item() * batch_size
        dataset_size += batch_size

        # Update progress bar
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    # Step scheduler if it's epoch-based (CosineAnnealingLR is usually stepped per epoch)
    if scheduler is not None:
        scheduler.step()

    epoch_loss = running_loss / dataset_size
    print(f"Train Loss: {epoch_loss:.16f}")

    return epoch_loss


def validate(model, dataloader, device):
    """
    Evaluates the model on the validation set (Loss only).
    """
    model.eval()
    criterion = ThoracicLoss()

    running_loss = 0.0
    dataset_size = 0

    pbar = tqdm(dataloader, desc="[Val]", leave=False)

    with torch.no_grad():
        for batch_idx, (images, targets, image_ids) in enumerate(pbar):
            batch_size = images.size(0)

            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)

            loss, _ = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    print(f"Val Loss:   {epoch_loss:.16f}")

    return epoch_loss


def _nms(heatmap, kernel=3):
    """
    Applies Max Pooling NMS to the heatmap.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def decode_predictions(outputs, image_ids, device, is_test=True):
    """
    Decodes model outputs into bounding boxes and class labels.
    Applies Gated Inference and Coupled Rescaling.
    """
    # Unpack outputs
    heatmap = outputs["heatmap"]  # (B, NumClasses-1, H, W)
    size = outputs["size"]  # (B, 2, H, W)
    offset = outputs["offset"]  # (B, 2, H, W)
    global_prob = outputs["global_prob"]  # (B, 1)

    batch_size, num_classes, h_out, w_out = heatmap.shape

    # Apply NMS
    heatmap = _nms(heatmap)

    predictions = []

    for b in range(batch_size):
        image_id = image_ids[b]

        # --- Gated Inference Check ---
        # global_prob is P(No Finding). If high, suppress everything.
        p_no_finding = global_prob[b].item()

        if p_no_finding > Config.NO_FINDING_PROB_THRESHOLD:
            # Explicit "No finding"
            predictions.append(
                {"image_id": image_id, "PredictionString": "14 1 0 0 1 1"}
            )
            continue

        # --- Extract Detections ---
        # Flatten for top-k or thresholding
        hm_flat = heatmap[b].view(-1)

        # Filter by confidence threshold
        mask = hm_flat > Config.CONF_THRESHOLD
        if not mask.any():
            # No findings detected above threshold
            predictions.append(
                {"image_id": image_id, "PredictionString": "14 1 0 0 1 1"}
            )
            continue

        indices = torch.nonzero(mask, as_tuple=False).squeeze(1)
        scores = hm_flat[indices]

        # Limit detections per image
        if len(scores) > Config.MAX_DETECTIONS_PER_IMAGE:
            scores, topk_idx = torch.topk(scores, Config.MAX_DETECTIONS_PER_IMAGE)
            indices = indices[topk_idx]

        # Convert indices to (c, y, x)
        # hm_flat is (C * H * W)
        # index = c * (H*W) + y * W + x
        area = h_out * w_out
        cls_ids = indices // area
        spatial_indices = indices % area
        ys = spatial_indices // w_out
        xs = spatial_indices % w_out

        # Get Size and Offset
        # size: (2, H, W), offset: (2, H, W)
        # We need to gather values at (ys, xs)

        # Helper to gather
        def gather_feat(feat, y, x):
            # feat: (2, H, W)
            # result: (N, 2)
            return feat[:, y, x].t()

        wh = gather_feat(size[b], ys, xs)  # (N, 2) -> w, h
        off = gather_feat(offset[b], ys, xs)  # (N, 2) -> x_off, y_off

        # Reconstruct Boxes (CenterNet logic)
        # Center = (x + off_x, y + off_y)
        # Box = Center +/- wh/2

        xs = xs.float() + off[:, 0]
        ys = ys.float() + off[:, 1]

        w = wh[:, 0]
        h = wh[:, 1]

        x_min = xs - w / 2
        y_min = ys - h / 2
        x_max = xs + w / 2
        y_max = ys + h / 2

        # --- Coupled Rescaling ---
        # 1. Scale from Output Map (160) to Input Size (640)
        scale_factor = Config.DOWNSAMPLE_RATIO
        x_min *= scale_factor
        y_min *= scale_factor
        x_max *= scale_factor
        y_max *= scale_factor

        # 2. Scale from Input Size (640) to Original Image Dimensions
        # We need to fetch original dimensions.
        # Construct path based on mode.
        if is_test:
            path = f"test/{image_id}.dicom"
        else:
            # For val, we assume standard train structure
            path = f"train/{image_id}.dicom"

        # Use the utility to get cached dimensions
        _, orig_h, orig_w = get_image_and_dimensions(
            image_id, path, load_cached_data=True
        )

        # Calculate ratios
        # Albumentations Resize keeps aspect ratio?
        # Config says A.Resize(Config.IMG_SIZE, Config.IMG_SIZE). This distorts aspect ratio if not square.
        # So we simply scale by ratio of dimensions.

        ratio_x = orig_w / Config.IMG_SIZE
        ratio_y = orig_h / Config.IMG_SIZE

        x_min *= ratio_x
        y_min *= ratio_y
        x_max *= ratio_x
        y_max *= ratio_y

        # Clip to image boundaries
        x_min = torch.clamp(x_min, 0, orig_w)
        y_min = torch.clamp(y_min, 0, orig_h)
        x_max = torch.clamp(x_max, 0, orig_w)
        y_max = torch.clamp(y_max, 0, orig_h)

        # Format Prediction String
        res_strings = []
        for i in range(len(scores)):
            cid = int(cls_ids[i].item())
            conf = float(scores[i].item())
            xmin = float(x_min[i].item())
            ymin = float(y_min[i].item())
            xmax = float(x_max[i].item())
            ymax = float(y_max[i].item())

            # Format: class_id confidence xmin ymin xmax ymax
            res_strings.append(
                f"{cid} {conf:.4f} {xmin:.1f} {ymin:.1f} {xmax:.1f} {ymax:.1f}"
            )

        if len(res_strings) == 0:
            predictions.append(
                {"image_id": image_id, "PredictionString": "14 1 0 0 1 1"}
            )
        else:
            predictions.append(
                {"image_id": image_id, "PredictionString": " ".join(res_strings)}
            )

    return predictions


def generate_submission(model, dataloader, device):
    """
    Runs inference on the test set and generates the submission file.
    """
    model.eval()
    all_preds = []

    print("Generating submission...")
    pbar = tqdm(dataloader, desc="[Inference]", leave=False)

    with torch.no_grad():
        for images, _, image_ids in pbar:
            images = images.to(device)

            # Forward
            outputs = model(images)

            # Decode
            batch_preds = decode_predictions(outputs, image_ids, device, is_test=True)
            all_preds.extend(batch_preds)

    # Create DataFrame
    df_sub = pd.DataFrame(all_preds)

    # Ensure columns order
    df_sub = df_sub[["image_id", "PredictionString"]]

    # Save
    save_path = Config.SUBMISSION_PATH
    df_sub.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

    return df_sub

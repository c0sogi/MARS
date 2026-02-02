import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from library.config import (
    DEVICE,
    SUBMISSION_FILE,
    IMG_SIZE,
    NUM_CLASSES,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
)
from library.model import EfficientDetDecoupled
from library.data import create_dataloaders


def decode_predictions(heatmap, size, offset, threshold=0.2, top_k=50):
    """
    Decodes the model outputs into bounding boxes.

    Args:
        heatmap (torch.Tensor): (B, C, H, W) logits.
        size (torch.Tensor): (B, 2, H, W) width/height.
        offset (torch.Tensor): (B, 2, H, W) x/y offsets.
        threshold (float): Score threshold.
        top_k (int): Max number of objects per image.

    Returns:
        list: List of tensors [batch_idx, class_id, score, x1, y1, x2, y2]
    """
    batch_size, _, height, width = heatmap.shape
    stride = IMG_SIZE // height  # Should be 4

    # 1. Heatmap Processing
    heatmap = torch.sigmoid(heatmap)

    # NMS via Max Pooling
    # Keep only pixels that are equal to the local max
    hmax = F.max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
    keep = (hmax == heatmap).float()
    heatmap = heatmap * keep

    # Top K selection
    # Flatten: (B, C, H, W) -> (B, C*H*W)
    heatmap_flat = heatmap.view(batch_size, -1)

    topk_scores, topk_inds = torch.topk(heatmap_flat, top_k)

    # Convert flattened index back to (c, y, x)
    topk_clses = torch.div(topk_inds, (height * width), rounding_mode="floor")
    topk_inds = topk_inds % (height * width)
    topk_ys = torch.div(topk_inds, width, rounding_mode="floor")
    topk_xs = topk_inds % width

    # Filter by threshold
    mask = topk_scores > threshold

    results = []

    for b in range(batch_size):
        b_mask = mask[b]
        if b_mask.sum() == 0:
            results.append(None)
            continue

        b_scores = topk_scores[b, b_mask]
        b_clses = topk_clses[b, b_mask]
        b_ys = topk_ys[b, b_mask]
        b_xs = topk_xs[b, b_mask]

        # Gather regression values
        # Size: (2, H, W), Offset: (2, H, W)
        # We need to select specific (y, x) locations

        # Prepare indices for gathering
        # We can just index directly since we are iterating batch
        b_size_w = size[b, 0, b_ys, b_xs]
        b_size_h = size[b, 1, b_ys, b_xs]
        b_off_x = offset[b, 0, b_ys, b_xs]
        b_off_y = offset[b, 1, b_ys, b_xs]

        # Reconstruct center in Input Scale (640x640)
        # Center = (Index + Offset) * Stride
        # Note: Offset is in feature map scale [0, 1]
        center_x = (b_xs.float() + b_off_x) * stride
        center_y = (b_ys.float() + b_off_y) * stride

        # Size is in Feature Map Scale, convert to Input Scale (Cite Lesson 40)
        w = b_size_w * stride
        h = b_size_h * stride

        # Convert to xmin, ymin, xmax, ymax
        x1 = center_x - w / 2
        y1 = center_y - h / 2
        x2 = center_x + w / 2
        y2 = center_y + h / 2

        # Clamp to image bounds
        x1 = x1.clamp(min=0, max=IMG_SIZE)
        y1 = y1.clamp(min=0, max=IMG_SIZE)
        x2 = x2.clamp(min=0, max=IMG_SIZE)
        y2 = y2.clamp(min=0, max=IMG_SIZE)

        # Stack: (N, 6) -> [class, score, x1, y1, x2, y2]
        # We don't need batch index here as we store in list
        detections = torch.stack([b_clses.float(), b_scores, x1, y1, x2, y2], dim=1)
        results.append(detections)

    return results


def predict_and_format(model, data_loader, device, threshold=0.2, gate_threshold=0.8):
    """
    Runs inference, applies global gating, rescales boxes, and formats predictions.

    Args:
        gate_threshold (float): If P(No Finding) > gate_threshold, output No Finding.
    """
    model.eval()
    submission_rows = []

    # No Finding string constant
    NO_FINDING_STR = "14 1 0 0 1 1"

    with torch.no_grad():
        for images, _, image_ids, original_shapes in data_loader:
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # 1. Global Gating
            # Global Head predicts logits for "Finding" (1).
            # P(Finding) = sigmoid(logits)
            # P(No Finding) = 1 - P(Finding)
            # If P(No Finding) > gate_threshold, we force empty.
            global_probs = torch.sigmoid(outputs["global_logits"]).squeeze(1)  # (B,)
            p_no_finding = 1.0 - global_probs

            # 2. Decode Boxes
            # Returns list of tensors (N_det, 6) or None
            batch_dets = decode_predictions(
                outputs["heatmap"],
                outputs["size"],
                outputs["offset"],
                threshold=threshold,
            )

            batch_size = images.size(0)

            for i in range(batch_size):
                img_id = image_ids[i]
                orig_h, orig_w = original_shapes[i]
                orig_h = orig_h.item()
                orig_w = orig_w.item()

                # Check Gate
                if p_no_finding[i] > gate_threshold:
                    submission_rows.append([img_id, NO_FINDING_STR])
                    continue

                dets = batch_dets[i]

                # Check if any boxes detected
                if dets is None or len(dets) == 0:
                    submission_rows.append([img_id, NO_FINDING_STR])
                    continue

                # 3. Rescale Boxes
                # Current coords are in [0, IMG_SIZE] (640)
                scale_x = orig_w / IMG_SIZE
                scale_y = orig_h / IMG_SIZE

                pred_strings = []

                # dets: [class, score, x1, y1, x2, y2]
                for det in dets:
                    cls_id = int(det[0])
                    score = float(det[1])
                    x1 = float(det[2]) * scale_x
                    y1 = float(det[3]) * scale_y
                    x2 = float(det[4]) * scale_x
                    y2 = float(det[5]) * scale_y

                    # Format: class_id confidence xmin ymin xmax ymax
                    pred_strings.append(
                        f"{cls_id} {score:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}"
                    )

                submission_rows.append([img_id, " ".join(pred_strings)])

    return submission_rows


def inference(checkpoint_path=None):
    """
    Main inference pipeline.
    """
    print(f"Starting Inference on {DEVICE}...")

    # 1. Setup Data
    # We only need test loader
    _, _, test_loader = create_dataloaders()

    if test_loader is None:
        print("No test data found. Skipping inference.")
        return

    # 2. Setup Model
    model = EfficientDetDecoupled(num_classes=NUM_CLASSES).to(DEVICE)

    # Load Checkpoint
    if checkpoint_path is None:
        # Default to best model
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        model.load_state_dict(state_dict)
    else:
        print(
            f"Warning: Checkpoint {checkpoint_path} not found. Using random weights (for debugging only)."
        )

    # 3. Run Prediction
    # Thresholds:
    # Detection Threshold: 0.2 (Keep low to maximize recall, mAP handles precision via conf scores)
    # Global Gate Threshold: 0.8 (Only suppress if very confident there is nothing)
    predictions = predict_and_format(
        model, test_loader, DEVICE, threshold=0.2, gate_threshold=0.8
    )

    # 4. Save Submission
    df_sub = pd.DataFrame(predictions, columns=["image_id", "PredictionString"])

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    df_sub.to_csv(SUBMISSION_FILE, index=False)

    print(f"Submission saved to {SUBMISSION_FILE}")
    print(df_sub.head())

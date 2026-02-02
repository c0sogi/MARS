import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.model import PointPillars
from library.dataset import NuScenesDataset
from library.utils import iou_2d


def decode_predictions(cls_preds, reg_preds, anchors):
    """
    Decodes model output into 3D bounding boxes.

    Args:
        cls_preds (torch.Tensor): (B, Num_Anchors*1, H, W)
        reg_preds (torch.Tensor): (B, Num_Anchors*7, H, W)
        anchors (torch.Tensor): (Total_Anchors, 7) [x, y, z, w, l, h, rot]

    Returns:
        batch_boxes (torch.Tensor): (B, Total_Anchors, 7)
        batch_scores (torch.Tensor): (B, Total_Anchors, Num_Classes)
    """
    batch_size = cls_preds.shape[0]

    # Permute to (B, H, W, Num_Anchors_Per_Loc, ...) to match anchor order
    # Anchors in dataset are flattened (H, W, NA, 7) -> (-1, 7)

    # 1. Process Classification Scores
    # (B, NA, H, W) -> (B, H, W, NA) -> (B, -1, 1)
    cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
    cls_preds = cls_preds.view(batch_size, -1, 1)

    # Apply sigmoid
    batch_scores = torch.sigmoid(cls_preds)

    # 2. Process Regression Offsets
    # (B, NA*7, H, W) -> (B, H, W, NA*7) -> (B, H, W, NA, 7) -> (B, -1, 7)
    num_anchors_per_loc = len(Config.CLASS_NAMES) * len(Config.ANCHOR_ROTATIONS)
    reg_preds = reg_preds.permute(0, 2, 3, 1).contiguous()
    reg_preds = reg_preds.view(batch_size, -1, 7)

    # 3. Apply Offsets to Anchors
    # Expand anchors to batch size
    anchors = anchors.unsqueeze(0).expand(batch_size, -1, -1)

    # Decode
    # anchors: x, y, z, w, l, h, rot
    # reg: dx, dy, dz, dw, dl, dh, drot

    # Diagonal of the base of the anchor
    d_a = torch.sqrt(anchors[..., 3] ** 2 + anchors[..., 4] ** 2)

    # x, y
    xs = reg_preds[..., 0] * d_a + anchors[..., 0]
    ys = reg_preds[..., 1] * d_a + anchors[..., 1]

    # z
    zs = reg_preds[..., 2] * anchors[..., 5] + anchors[..., 2]

    # w, l, h
    ws = torch.exp(reg_preds[..., 3]) * anchors[..., 3]
    ls = torch.exp(reg_preds[..., 4]) * anchors[..., 4]
    hs = torch.exp(reg_preds[..., 5]) * anchors[..., 5]

    # rot
    rots = reg_preds[..., 6] + anchors[..., 6]

    batch_boxes = torch.stack([xs, ys, zs, ws, ls, hs, rots], dim=-1)

    return batch_boxes, batch_scores


def nms_process(boxes, scores, score_thresh, iou_thresh, max_proposals):
    """
    Applies NMS to a single sample's predictions.

    Args:
        boxes (torch.Tensor): (N, 7)
        scores (torch.Tensor): (N, 1)
        score_thresh (float): Minimum score to keep.
        iou_thresh (float): IoU threshold for suppression.
        max_proposals (int): Max boxes to return.

    Returns:
        kept_boxes (torch.Tensor)
        kept_scores (torch.Tensor)
        kept_labels (torch.Tensor)
    """
    # 1. Filter by score
    scores = scores.squeeze(-1)
    mask = scores > score_thresh

    if not mask.any():
        return None, None, None

    boxes = boxes[mask]
    scores = scores[mask]

    # 2. Determine Class Labels based on Anchor Index
    # The anchors are generated in a specific order:
    # For each grid cell: Loop Class -> Loop Rotation
    # We need to recover the class index from the original index,
    # but here we just have the filtered list.
    # However, the `boxes` tensor corresponds to specific anchors.
    # We need the original indices to map back to classes.
    # Alternatively, we can assume the anchors passed to this function
    # are aligned with the flattened structure where we know the pattern.

    # Ideally, we should have passed class indices.
    # Let's reconstruct indices.
    original_indices = torch.nonzero(mask, as_tuple=False).squeeze(1)

    num_types = len(Config.CLASS_NAMES) * len(Config.ANCHOR_ROTATIONS)
    # The anchors are flattened (H * W * Num_Types)
    # But wait, dataset.py: stack(..., axis=-2).reshape(-1, 7)
    # This means the inner-most dimension is Num_Types.
    # So index % Num_Types gives the type index.

    type_indices = original_indices % num_types
    class_indices = type_indices // len(Config.ANCHOR_ROTATIONS)

    # 3. Sort by score
    sorted_indices = torch.argsort(scores, descending=True)
    boxes = boxes[sorted_indices]
    scores = scores[sorted_indices]
    class_indices = class_indices[sorted_indices]

    # 4. NMS
    keep = []

    # We use 2D IoU (BEV) for NMS as per standard practice
    # boxes: x, y, z, w, l, h, rot
    # iou_2d expects: x, y, w, l
    bev_boxes = boxes[:, [0, 1, 3, 4]]

    while bev_boxes.size(0) > 0:
        if len(keep) >= max_proposals:
            break

        # Pick the best
        keep.append(sorted_indices[0])  # Store original index if needed, or just count

        if bev_boxes.size(0) == 1:
            break

        current_box = bev_boxes[0:1]
        rest_boxes = bev_boxes[1:]

        # Calculate IoU
        ious = iou_2d(current_box, rest_boxes).squeeze(0)

        # Keep boxes with IoU < threshold
        valid_mask = ious < iou_thresh

        bev_boxes = rest_boxes[valid_mask]
        sorted_indices = sorted_indices[1:][valid_mask]  # Update indices tracking

        # Update other tensors for the loop
        # (We actually just need to slice the original arrays to build the result at the end
        #  but doing it iteratively is slow in python.
        #  Better to collect indices of the 'boxes' tensor we created in step 3)

    # Re-implement NMS efficiently using indices
    # Since we can't use torchvision.ops.nms (no rotation support in standard,
    # and we are using axis aligned approximation for NMS here based on iou_2d utils)

    # Let's restart the NMS part with a simpler index list approach
    keep_indices = []
    candidates_idx = torch.arange(boxes.size(0), device=boxes.device)

    # Limit candidates for speed
    if boxes.size(0) > 4000:
        candidates_idx = candidates_idx[:4000]

    while len(candidates_idx) > 0:
        if len(keep_indices) >= max_proposals:
            break

        current = candidates_idx[0]
        keep_indices.append(current.item())

        if len(candidates_idx) == 1:
            break

        current_box = boxes[current, [0, 1, 3, 4]].unsqueeze(0)
        others_idx = candidates_idx[1:]
        other_boxes = boxes[others_idx, :][:, [0, 1, 3, 4]]

        ious = iou_2d(current_box, other_boxes).squeeze(0)

        # Keep those with low IoU
        keep_mask = ious < iou_thresh
        candidates_idx = others_idx[keep_mask]

    keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)

    return boxes[keep_indices], scores[keep_indices], class_indices[keep_indices]


def format_submission(sample_token, boxes, scores, class_indices):
    """
    Formats predictions into the submission string.
    """
    if boxes is None or len(boxes) == 0:
        return f"{sample_token},"

    pred_strings = []

    # Move to CPU
    boxes = boxes.cpu().numpy()
    scores = scores.cpu().numpy()
    class_indices = class_indices.cpu().numpy()

    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        cls_idx = class_indices[i]
        cls_name = Config.CLASS_NAMES[cls_idx]

        # Format: confidence x y z w l h yaw class_name
        # box: x, y, z, w, l, h, rot
        s = f"{score:.4f} {box[0]:.4f} {box[1]:.4f} {box[2]:.4f} {box[3]:.4f} {box[4]:.4f} {box[5]:.4f} {box[6]:.4f} {cls_name}"
        pred_strings.append(s)

    prediction_string = " ".join(pred_strings)
    return f"{sample_token},{prediction_string}"


def generate_submission(
    model_path=Config.MODEL_SAVE_PATH, output_path=Config.SUBMISSION_PATH
):
    """
    Generates the submission file for the test set.
    """
    print(f"Generating submission from model: {model_path}")

    # 1. Setup
    device = torch.device(Config.DEVICE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 2. Load Data
    # We rely on NuScenesDataset to handle metadata loading and caching
    test_dataset = NuScenesDataset(split="test", load_cached_data=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
    )

    # 3. Load Model
    model = PointPillars().to(device)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print(
            f"Warning: Model checkpoint not found at {model_path}. Using random weights."
        )

    model.eval()

    # 4. Get Anchors
    # We need anchors on the device
    anchors = torch.from_numpy(test_dataset.anchors).to(device)

    results = []

    print("Running inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            # Move data to device
            batch_pillars = batch["pillars"].to(device)
            batch_coors = batch["coors"].to(device)
            batch_n_points = batch["n_points"].to(device)
            sample_tokens = batch["sample_tokens"]

            input_dict = {
                "pillars": batch_pillars,
                "coors": batch_coors,
                "n_points": batch_n_points,
                "sample_tokens": sample_tokens,
            }

            # Forward
            output = model(input_dict)
            cls_preds = output["cls_preds"]
            reg_preds = output["reg_preds"]

            # Decode
            batch_boxes, batch_scores = decode_predictions(
                cls_preds, reg_preds, anchors
            )

            # Post-process per sample
            for i in range(len(sample_tokens)):
                token = sample_tokens[i]
                boxes = batch_boxes[i]
                scores = batch_scores[i]

                # NMS
                final_boxes, final_scores, final_classes = nms_process(
                    boxes,
                    scores,
                    score_thresh=Config.NMS_SCORE_THRESHOLD,
                    iou_thresh=Config.NMS_IOU_THRESHOLD,
                    max_proposals=Config.MAX_PROPOSALS,
                )

                # Format
                line = format_submission(
                    token, final_boxes, final_scores, final_classes
                )
                results.append(line)

    # 5. Write to CSV
    print(f"Writing {len(results)} predictions to {output_path}")
    with open(output_path, "w") as f:
        f.write("Id,PredictionString\n")
        for line in results:
            f.write(line + "\n")

    print("Submission generation complete.")


if __name__ == "__main__":
    # This block is not required by the prompt instructions but facilitates testing
    pass

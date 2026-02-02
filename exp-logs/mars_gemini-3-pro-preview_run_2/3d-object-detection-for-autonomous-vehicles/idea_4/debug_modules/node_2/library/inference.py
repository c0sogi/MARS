import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from collections import defaultdict
from torch.utils.data import DataLoader

from library.config import Config
from library.model import MonoCenterNet
from library.dataset import Mono3DDataset
import library.utils as utils
from library.loss import _transpose_and_gather_feat


def _nms(heatmap, kernel=3):
    """
    Applies Max-Pooling NMS to the heatmap.
    """
    pad = (kernel - 1) // 2
    hmax = F.max_pool2d(heatmap, (kernel, kernel), stride=1, padding=pad)
    keep = (hmax == heatmap).float()
    return heatmap * keep


def _topk(scores, K=40):
    """
    Extracts top K scores and their indices from the heatmap.
    Args:
        scores: (B, C, H, W)
        K: int
    Returns:
        topk_score: (B, K)
        topk_inds: (B, K) flattened indices in H*W
        topk_clses: (B, K) class indices
        topk_ys: (B, K) y coordinates
        topk_xs: (B, K) x coordinates
    """
    batch, cat, height, width = scores.size()

    # (B, C, H*W)
    topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

    topk_inds = topk_inds % (height * width)
    topk_ys = (topk_inds / width).int().float()
    topk_xs = (topk_inds % width).int().float()

    # (B, K)
    topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
    topk_clses = (topk_ind / K).int()

    # Gather spatial indices based on the top K across all classes
    topk_inds = _gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_ys = _gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
    topk_xs = _gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

    return topk_score, topk_inds, topk_clses, topk_ys, topk_xs


def _gather_feat(feat, ind):
    """
    Gather feature from specified indices.
    feat: (B, N, C)
    ind: (B, K)
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    return feat


def decode_detections(outputs, info, K=50, conf_thresh=0.2):
    """
    Decodes model outputs into 3D bounding boxes in Global Frame.
    Args:
        outputs: Dict of tensors (hm, depth, dim, rot, offset)
        info: Dict containing calibration info (batch_size=1 expected for simplicity)
        K: Top K objects to keep
        conf_thresh: Confidence threshold
    Returns:
        results: List of prediction strings for the sample
    """
    hm = torch.sigmoid(outputs["hm"])
    hm = _nms(hm)

    batch_size = hm.size(0)
    # We assume batch_size = 1 for inference to handle varying intrinsics easily

    scores, inds, clses, ys, xs = _topk(hm, K=K)

    # Gather regression heads
    # outputs are (B, C, H, W), _transpose_and_gather expects (B, C, H, W) and (B, K)
    # returns (B, K, C)
    depth = _transpose_and_gather_feat(outputs["depth"], inds)
    dim = _transpose_and_gather_feat(outputs["dim"], inds)
    rot = _transpose_and_gather_feat(outputs["rot"], inds)
    off = _transpose_and_gather_feat(outputs["offset"], inds)

    # Convert to numpy for geometric processing
    scores = scores.cpu().numpy()
    clses = clses.cpu().numpy()
    ys = ys.cpu().numpy()
    xs = xs.cpu().numpy()
    depth = depth.cpu().numpy()
    dim = dim.cpu().numpy()
    rot = rot.cpu().numpy()
    off = off.cpu().numpy()

    predictions = []

    for b in range(batch_size):
        # Extract scalar/array info for this batch item
        # Since dataloader collates into tensors, we convert back to numpy
        intrinsics = info["intrinsics"][b].numpy()
        ego_t = info["ego_translation"][b].numpy()
        ego_r = info["ego_rotation"][b].numpy()
        sensor_t = info["sensor_translation"][b].numpy()
        sensor_r = info["sensor_rotation"][b].numpy()
        cam_yaw = info["cam_yaw"][b].item()

        # Camera parameters
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]

        # Stride correction
        stride = Config.DOWN_RATIO

        for i in range(K):
            score = scores[b, i]
            if score < conf_thresh:
                continue

            cls_id = int(clses[b, i])
            class_name = Config.CLASS_NAMES[cls_id]

            # 1. Recover 2D center
            # Add predicted offset and scale by stride
            u = (xs[b, i] + off[b, i, 0]) * stride
            v = (ys[b, i] + off[b, i, 1]) * stride

            # 2. Recover Depth
            z_cam = depth[b, i, 0]

            # 3. Back-project to 3D Camera Coordinates
            # x = (u - cx) * z / fx
            x_cam = (u - cx) * z_cam / fx
            y_cam = (v - cy) * z_cam / fy

            # Point in camera frame
            pt_cam = np.array([[x_cam, y_cam, z_cam]])

            # 4. Transform to Global Frame
            pt_global = utils.camera_to_global(pt_cam, ego_t, ego_r, sensor_t, sensor_r)
            center_x, center_y, center_z = pt_global[0]

            # 5. Dimensions
            width, length, height = dim[b, i]

            # 6. Yaw
            # Local yaw from sin, cos
            sin_r, cos_r = rot[b, i]
            local_yaw = np.arctan2(sin_r, cos_r)
            # Global yaw
            yaw = local_yaw + cam_yaw

            # Format: confidence x y z width length height yaw class_name
            pred_str = f"{score:.4f} {center_x:.4f} {center_y:.4f} {center_z:.4f} {width:.4f} {length:.4f} {height:.4f} {yaw:.4f} {class_name}"
            predictions.append(pred_str)

    return predictions


def generate_submission(
    checkpoint_path, split="test", debug=False, load_cached_data=True
):
    """
    Runs inference on the test set and generates the submission file.
    """
    Config.setup()
    device = Config.DEVICE

    print(f"Loading model from {checkpoint_path}...")
    model = MonoCenterNet()

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}. Cannot generate submission.")
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    # Prepare Dataset
    # We use batch_size=1 to simplify handling of varying intrinsics/extrinsics
    dataset = Mono3DDataset(split=split, load_cached_data=load_cached_data, debug=debug)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    results = defaultdict(list)

    print("Starting Inference...")

    with torch.no_grad():
        for i, (img, _, info) in enumerate(dataloader):
            img = img.to(device)

            # Forward pass
            outputs = model(img)

            # Decode
            preds = decode_detections(
                outputs, info, K=Config.TOP_K, conf_thresh=Config.CONF_THRESHOLD
            )

            # Aggregate by sample token
            # info['token'] is a list (batch size 1)
            sample_token = info["token"][0]
            results[sample_token].extend(preds)

    # Format Submission
    print("Formatting submission...")

    # Load sample submission to ensure we have all IDs and correct order
    sample_sub_path = os.path.join(Config.INPUT_DIR, "sample_submission.csv")
    df_sub = pd.read_csv(sample_sub_path)

    final_preds = []
    for token in df_sub["Id"]:
        if token in results and len(results[token]) > 0:
            # Join all predictions for this sample with a space
            pred_string = " ".join(results[token])
        else:
            pred_string = ""  # Empty prediction

        final_preds.append(pred_string)

    df_sub["PredictionString"] = final_preds

    out_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    df_sub.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

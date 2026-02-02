import os
import random
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import library.config as config
import library.utils as utils
from library.data_interface import DataInterface
from library.dataset import BEVDataset
from library.model import BEVDetector


def _gather_feat(feat, ind, mask=None):
    """
    Gather features from a feature map at specific indices.
    feat: (B, C, H, W) -> flattened to (B, H*W, C)
    ind: (B, K) indices
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    """
    Transpose feature map and gather features.
    feat: (B, C, H, W)
    ind: (B, K)
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


def decode_predictions(hm, reg, K=100, threshold=0.1):
    """
    Decodes model outputs into 3D bounding boxes in the Sensor Frame.

    Args:
        hm (torch.Tensor): Heatmap logits (B, NumClasses, H, W).
        reg (torch.Tensor): Regression map (B, 8, H, W).
        K (int): Top K peaks to keep per sample.
        threshold (float): Confidence threshold.

    Returns:
        list of dicts: Per-sample predictions containing:
            'bboxes': (N, 7) [x, y, z, w, l, h, yaw]
            'scores': (N,)
            'labels': (N,)
    """
    batch_size, num_classes, H, W = hm.size()

    # 1. Heatmap processing
    hm = torch.sigmoid(hm)

    # 3x3 Max Pooling (NMS)
    hmax = F.max_pool2d(hm, kernel_size=3, padding=1, stride=1)
    keep = (hmax == hm).float()
    hm = hm * keep

    # 2. Top K selection
    # Flatten: (B, C, H, W) -> (B, C * H * W)
    hm_flat = hm.view(batch_size, -1)
    topk_scores, topk_inds = torch.topk(hm_flat, K)

    # Convert flattened index to (cl, y, x)
    topk_cl = (topk_inds // (H * W)).int()
    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds // W).int().float()
    topk_xs = (topk_inds % W).int().float()

    # 3. Gather Regression Values
    # reg: (B, 8, H, W)
    # Channels: 0:ox, 1:oy, 2:z, 3:lw, 4:ll, 5:lh, 6:sin, 7:cos
    reg = _transpose_and_gather_feat(reg, topk_inds)  # (B, K, 8)

    off_x = reg[..., 0]
    off_y = reg[..., 1]
    z = reg[..., 2]
    w = torch.exp(reg[..., 3])
    l = torch.exp(reg[..., 4])
    h = torch.exp(reg[..., 5])
    sin_yaw = reg[..., 6]
    cos_yaw = reg[..., 7]
    yaw = torch.atan2(sin_yaw, cos_yaw)

    # 4. Coordinate Conversion (Grid -> Metric Sensor Frame)
    # x_sensor = (x_grid + off_x) * stride * voxel_size + min_range
    stride = config.DOWN_RATIO
    voxel_x = config.VOXEL_SIZE[0]
    voxel_y = config.VOXEL_SIZE[1]
    min_x = config.POINT_CLOUD_RANGE[0]
    min_y = config.POINT_CLOUD_RANGE[1]

    xs = (topk_xs + off_x) * stride * voxel_x + min_x
    ys = (topk_ys + off_y) * stride * voxel_y + min_y

    # Stack: (B, K, 7)
    bboxes = torch.stack([xs, ys, z, w, l, h, yaw], dim=2)
    scores = topk_scores
    labels = topk_cl

    # 5. Filter by Threshold and Format
    results = []
    for i in range(batch_size):
        mask = scores[i] > threshold

        sample_boxes = bboxes[i][mask].cpu().numpy()
        sample_scores = scores[i][mask].cpu().numpy()
        sample_labels = labels[i][mask].cpu().numpy()

        results.append(
            {"bboxes": sample_boxes, "scores": sample_scores, "labels": sample_labels}
        )

    return results


def transform_predictions_to_world(predictions, sample_tokens, data_interface):
    """
    Transforms predictions from Sensor Frame to World Frame.

    Args:
        predictions (list): List of dicts from decode_predictions.
        sample_tokens (list): List of sample tokens corresponding to the batch.
        data_interface (DataInterface): Interface to get transform matrices.

    Returns:
        list of dicts: Predictions with 'bboxes' in World Frame.
    """
    transformed_preds = []

    for i, pred in enumerate(predictions):
        token = sample_tokens[i]
        boxes = pred["bboxes"]  # (N, 7) [x, y, z, w, l, h, yaw]

        if len(boxes) == 0:
            transformed_preds.append(pred)
            continue

        # Get Transform: World -> Sensor
        try:
            world_to_sensor = data_interface.get_transform_matrix(token)
            # Invert to get Sensor -> World
            sensor_to_world = np.linalg.inv(world_to_sensor)
        except Exception as e:
            print(f"Error getting transform for {token}: {e}")
            transformed_preds.append(pred)
            continue

        # 1. Transform Centers (x, y, z)
        centers_sensor = boxes[:, :3]
        centers_world = utils.transform_points(centers_sensor, sensor_to_world)

        # 2. Transform Yaw
        # Rotate unit vector [cos(yaw), sin(yaw), 0]
        yaws_sensor = boxes[:, 6]
        c_yaw = np.cos(yaws_sensor)
        s_yaw = np.sin(yaws_sensor)
        zeros = np.zeros_like(yaws_sensor)
        vec_sensor = np.stack([c_yaw, s_yaw, zeros], axis=1)  # (N, 3)

        # Apply rotation only (top-left 3x3)
        R = sensor_to_world[:3, :3]
        # (R @ v.T).T -> v @ R.T
        vec_world = vec_sensor @ R.T

        yaws_world = np.arctan2(vec_world[:, 1], vec_world[:, 0])

        # 3. Reassemble
        # Dimensions (w, l, h) [indices 3, 4, 5] do not change
        boxes_world = np.copy(boxes)
        boxes_world[:, :3] = centers_world
        boxes_world[:, 6] = yaws_world

        transformed_preds.append(
            {"bboxes": boxes_world, "scores": pred["scores"], "labels": pred["labels"]}
        )

    return transformed_preds


def format_prediction_string(prediction):
    """
    Formats a single sample's predictions into the submission string format.
    Format: confidence x y z w l h yaw class_name ...
    """
    boxes = prediction["bboxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    if len(boxes) == 0:
        return ""

    parts = []
    for i in range(len(boxes)):
        box = boxes[i]
        score = scores[i]
        label_id = int(labels[i])
        class_name = config.ID_TO_CLASS[label_id]

        # box: [x, y, z, w, l, h, yaw]
        # Submission format expects: confidence cx cy cz w l h yaw class_name
        # Note: box[3]=w, box[4]=l, box[5]=h

        # Ensure values are standard floats
        s_str = f"{score:.4f}"
        x_str = f"{box[0]:.4f}"
        y_str = f"{box[1]:.4f}"
        z_str = f"{box[2]:.4f}"
        w_str = f"{box[3]:.4f}"
        l_str = f"{box[4]:.4f}"
        h_str = f"{box[5]:.4f}"
        yaw_str = f"{box[6]:.4f}"

        parts.append(
            f"{s_str} {x_str} {y_str} {z_str} {w_str} {l_str} {h_str} {yaw_str} {class_name}"
        )

    return " ".join(parts)


def generate_submission(
    model_path,
    batch_size=config.BATCH_SIZE,
    load_cached_data=True,
    num_workers=config.NUM_WORKERS,
    threshold=0.1,
):
    """
    Main inference loop to generate submission.csv.
    """
    logger = utils.get_logger(__name__)
    logger.info("Starting Submission Generation...")

    # 1. Setup
    config.set_seed(config.SEED)
    device = config.get_device()

    # 2. Data Interface
    data_interface = DataInterface(load_cached_data=load_cached_data)

    # 3. Test Dataset
    test_dataset = BEVDataset(
        split="test", data_interface=data_interface, load_cached_data=load_cached_data
    )

    # Reproducibility for DataLoader (Cite solution_lesson_node_00010)
    g = torch.Generator()
    g.manual_seed(config.SEED)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # 4. Load Model
    model = BEVDetector(
        num_classes=config.NUM_CLASSES,
        backbone_name=config.BACKBONE,
        pretrained=False,  # No need to download weights, loading state dict
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    logger.info(f"Loading model weights from {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 5. Inference Loop
    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, disable=True):  # Silent execution as requested
            inputs = batch["input"].to(device)
            sample_tokens = batch["sample_token"]  # List of strings

            # Forward
            hm, reg = model(inputs)

            # Decode (Sensor Frame)
            batch_preds_sensor = decode_predictions(hm, reg, threshold=threshold)

            # Transform (World Frame)
            batch_preds_world = transform_predictions_to_world(
                batch_preds_sensor, sample_tokens, data_interface
            )

            # Format
            for i, token in enumerate(sample_tokens):
                pred_str = format_prediction_string(batch_preds_world[i])
                results.append({"Id": token, "PredictionString": pred_str})

    # 6. Save Submission
    df = pd.DataFrame(results)

    # Ensure all test IDs are present (fill missing with empty if any)
    # The dataset iteration should cover all, but good practice to check against sample_submission
    sample_sub_path = os.path.join(config.INPUT_DIR, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        sample_df = pd.read_csv(sample_sub_path)
        # Merge to ensure order and completeness
        final_df = sample_df[["Id"]].merge(df, on="Id", how="left")
        final_df["PredictionString"] = final_df["PredictionString"].fillna("")
    else:
        final_df = df

    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    final_df.to_csv(config.SUBMISSION_PATH, index=False)
    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

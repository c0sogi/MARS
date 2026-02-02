import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import NuScenesDataset
from library.model import PillarUNet3D
from library.utils import transform_box_to_global


def decode_predictions(preds, config):
    """
    Decodes model output (heatmap + regression) into 3D bounding boxes in Ego frame.

    Args:
        preds (dict): Dictionary containing 'hm' and 'reg' tensors.
        config (class): Configuration class.

    Returns:
        list of lists: A list where each element is a list of dictionaries
                       {'box': [x, y, z, w, l, h, yaw], 'score': float, 'class_name': str}
                       for a sample in the batch.
    """
    hm = preds["hm"]  # (B, C, H, W)
    reg = preds["reg"]  # (B, 8, H, W)
    batch_size = hm.shape[0]

    # 1. Heatmap Processing (Sigmoid + NMS via MaxPool)
    hm = torch.sigmoid(hm)

    # 3x3 Max Pooling to find local peaks
    padding = (1, 1, 1, 1)
    hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=1)
    keep = (hmax == hm).float()
    hm = hm * keep  # Zero out non-maxima

    # 2. Extract Top K Peaks
    # Flatten spatial dimensions: (B, C, H*W)
    scores = hm.reshape(batch_size, -1)
    topk_scores, topk_inds = torch.topk(scores, config.MAX_DETECTIONS)

    # Convert flattened indices to (class, y, x)
    C = config.NUM_CLASSES
    H, W = config.GRID_SIZE[1], config.GRID_SIZE[0]

    # topk_inds is index in C*H*W
    topk_cl = (topk_inds // (H * W)).int()
    topk_inds = topk_inds % (H * W)
    topk_ys = (topk_inds // W).int()
    topk_xs = (topk_inds % W).int()

    # 3. Gather Regression Targets
    # reg is (B, 8, H, W) -> permute to (B, H, W, 8) -> view (B, H*W, 8)
    # We gather features at the spatial locations of the peaks
    reg = reg.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 8)

    # Expand indices for gathering: (B, K) -> (B, K, 8)
    gather_inds = topk_inds.unsqueeze(2).expand(batch_size, config.MAX_DETECTIONS, 8)
    topk_reg = reg.gather(1, gather_inds)

    results = []

    for b in range(batch_size):
        sample_boxes = []

        # Filter by confidence threshold
        mask = topk_scores[b] > config.CONFIDENCE_THRESHOLD

        if not mask.any():
            results.append([])
            continue

        valid_scores = topk_scores[b][mask]
        valid_cl = topk_cl[b][mask]
        valid_reg = topk_reg[b][mask]
        valid_xs = topk_xs[b][mask]
        valid_ys = topk_ys[b][mask]

        for i in range(len(valid_scores)):
            # Decode Regression Values
            # Target: [off_x, off_y, z, log_w, log_l, log_h, sin, cos]
            off_x, off_y, z, log_w, log_l, log_h, sin_y, cos_y = valid_reg[i]

            # 1. Center Coordinates (Grid -> Ego)
            # x_grid = x_idx + offset
            # x_ego = x_grid * voxel_size + min_range
            xs = valid_xs[i].float() + off_x
            ys = valid_ys[i].float() + off_y

            x_ego = xs * config.VOXEL_SIZE[0] + config.POINT_CLOUD_RANGE[0]
            y_ego = ys * config.VOXEL_SIZE[1] + config.POINT_CLOUD_RANGE[1]

            # 2. Dimensions (Log space -> Linear)
            w = torch.exp(log_w)
            l = torch.exp(log_l)
            h = torch.exp(log_h)

            # 3. Yaw (Sin/Cos -> Radians)
            yaw = torch.atan2(sin_y, cos_y)

            score = valid_scores[i].item()
            cls_id = valid_cl[i].item()
            cls_name = config.CLASS_NAMES[cls_id]

            # Box format: [x, y, z, w, l, h, yaw]
            box = np.array(
                [
                    x_ego.item(),
                    y_ego.item(),
                    z.item(),
                    w.item(),
                    l.item(),
                    h.item(),
                    yaw.item(),
                ],
                dtype=np.float32,
            )

            sample_boxes.append({"box": box, "score": score, "class_name": cls_name})

        results.append(sample_boxes)

    return results


def generate_submission(checkpoint_path, output_path=None):
    """
    Runs inference on the test set and generates the submission CSV.

    Args:
        checkpoint_path (str): Path to the model checkpoint (.pth).
        output_path (str, optional): Path to save submission.csv. Defaults to Config.SUBMISSION_PATH.
    """
    if output_path is None:
        output_path = Config.SUBMISSION_PATH

    # 1. Setup Data
    # Hack: Point VAL_METADATA_PATH to TEST_METADATA_PATH to reuse NuScenesDataset logic
    # The dataset class uses 'is_train=False' to load VAL_METADATA_PATH.
    original_val_path = Config.VAL_METADATA_PATH
    Config.VAL_METADATA_PATH = Config.TEST_METADATA_PATH

    try:
        dataset = NuScenesDataset(is_train=False, load_cached_data=True)
    finally:
        # Restore config just in case
        Config.VAL_METADATA_PATH = original_val_path

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=NuScenesDataset.collate_fn,
    )

    # 2. Setup Model
    device = Config.DEVICE
    model = PillarUNet3D().to(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    # Handle state dict key mismatch if wrapped in DataParallel/DDP
    state_dict = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    print(f"Starting inference on {len(dataset)} samples...")

    submission_rows = []

    with torch.no_grad():
        for batch in dataloader:
            # Move data to device
            batch["voxels"] = batch["voxels"].to(device)
            batch["num_points"] = batch["num_points"].to(device)
            batch["coordinates"] = batch["coordinates"].to(device)

            # Forward Pass
            preds = model(batch)

            # Decode to Ego Frame
            decoded_batch = decode_predictions(preds, Config)

            sample_tokens = batch["sample_token"]

            # Process each sample in batch
            for i, detections in enumerate(decoded_batch):
                token = sample_tokens[i]
                prediction_strings = []

                if len(detections) > 0:
                    # Sort by confidence descending
                    detections.sort(key=lambda x: x["score"], reverse=True)

                    # Get Ego Pose for Transformation
                    # We need to access the dataset's internal lookup tables
                    # 1. Find LIDAR_TOP sample_data token
                    lidar_token = dataset.sample_to_lidar_token.get(token)

                    if lidar_token:
                        # 2. Get Ego Pose
                        sd_rec = dataset.sd_lookup[lidar_token]
                        ep_token = sd_rec["ego_pose_token"]
                        ego_trans, ego_rot = dataset.get_pose(
                            ep_token, dataset.ep_lookup
                        )

                        for det in detections:
                            box_ego = det["box"]
                            score = det["score"]
                            cls_name = det["class_name"]

                            # Transform Ego -> Global
                            box_global = transform_box_to_global(
                                box_ego, ego_trans, ego_rot
                            )

                            # Format: confidence x y z w l h yaw class_name
                            # box_global is [x, y, z, w, l, h, yaw]
                            pred_str = (
                                f"{score:.4f} {box_global[0]:.4f} {box_global[1]:.4f} {box_global[2]:.4f} "
                                f"{box_global[3]:.4f} {box_global[4]:.4f} {box_global[5]:.4f} "
                                f"{box_global[6]:.4f} {cls_name}"
                            )

                            prediction_strings.append(pred_str)
                    else:
                        # Fallback if lidar token mapping missing (should not happen in valid test set)
                        pass

                # Join all predictions for this image
                full_pred_str = (
                    " ".join(prediction_strings) if prediction_strings else ""
                )

                # Append to result list
                # Use 'nan' for empty string if pandas requires, but usually empty string is fine.
                # The sample submission uses 'nan' for empty?
                # The sample submission snippet showed 'nan' in the table representation but likely empty string or NaN value.
                # We will use empty string.
                submission_rows.append({"Id": token, "PredictionString": full_pred_str})

    # 3. Save Submission
    df_sub = pd.DataFrame(submission_rows)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

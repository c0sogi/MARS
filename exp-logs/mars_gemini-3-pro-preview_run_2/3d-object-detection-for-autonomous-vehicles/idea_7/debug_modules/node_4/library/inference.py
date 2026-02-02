import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from library.config import Config
from library.modules import IoUAwareCenterPoint
from library.utils import transform_points


def load_model(checkpoint_path, device=Config.DEVICE):
    """
    Initializes the model and loads weights from the checkpoint.

    Args:
        checkpoint_path (str): Path to the model checkpoint.
        device (str): Device to load the model onto.

    Returns:
        model (nn.Module): The loaded model in eval mode.
    """
    model = IoUAwareCenterPoint().to(device)

    if os.path.exists(checkpoint_path):
        # Load state dict
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        # If checkpoint is missing, we return the initialized model (untrained)
        # This allows the pipeline to run even if training failed, though results will be poor.
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using random weights."
        )

    model.eval()
    return model


def predict_and_format(model, dataloader, output_path, device=Config.DEVICE):
    """
    Runs inference, rectifies scores with IoU, transforms coordinates,
    and generates the submission CSV.

    Args:
        model (nn.Module): Trained model.
        dataloader (DataLoader): Test dataloader.
        output_path (str): Path to save the submission CSV.
        device (str): Device to run inference on.
    """
    model.eval()

    # Ensure dataset cache is available for coordinate transforms
    dataset = dataloader.dataset
    if not hasattr(dataset, "lookup_table"):
        # Force load/build cache if not already present
        dataset._load_or_build_cache(True)

    # Create lookup index for fast retrieval of transforms
    # lookup_table columns: token, world_to_sensor, sweep_paths, sweep_transforms
    lookup = dataset.lookup_table.set_index("token")

    results = []

    # Grid parameters for decoding
    # Voxel size in BEV after downsampling
    voxel_x = Config.VOXEL_SIZE[0] * Config.DOWN_RATIO
    voxel_y = Config.VOXEL_SIZE[1] * Config.DOWN_RATIO

    # Point cloud range offsets
    pc_range_x = Config.POINT_CLOUD_RANGE[0]
    pc_range_y = Config.POINT_CLOUD_RANGE[1]

    # Hyperparameters
    alpha = Config.IOU_RECTIFIER_ALPHA
    conf_threshold = Config.CONF_THRESHOLD
    topk = Config.TOP_K

    with torch.no_grad():
        # Iterate without progress bar
        for batch in dataloader:
            points = [p.to(device) for p in batch["points"]]
            tokens = batch["metadata"]["tokens"]

            # Forward Pass
            preds = model({"points": points})

            # --- Decoding ---
            # Heatmap
            hm = preds["hm"]  # (B, C, H, W)
            hm = torch.sigmoid(hm)

            # NMS via MaxPool (3x3 kernel)
            pad = 1
            hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=pad)
            keep = (hmax == hm).float()
            hm = hm * keep

            # Top K selection
            B, C, H, W = hm.shape

            # Flatten spatial dims
            hm = hm.view(B, -1)
            scores, inds = torch.topk(hm, topk)

            # Convert flattened index to class and spatial index
            clses = inds // (H * W)
            inds = inds % (H * W)

            # Helper to gather features at specific indices
            def gather_feat(feat):
                # Permute to (B, H, W, C) then view as (B, H*W, C)
                feat = feat.permute(0, 2, 3, 1).contiguous()
                feat = feat.view(B, -1, feat.size(3))
                dim = feat.size(2)
                # Expand indices to match feature dimension
                ind_g = inds.unsqueeze(2).expand(B, topk, dim)
                return feat.gather(1, ind_g)

            # Gather regression heads
            reg = gather_feat(preds["reg"])
            wh = gather_feat(preds["wh"])
            rot = gather_feat(preds["rot"])
            z = gather_feat(preds["z"])
            iou = gather_feat(preds["iou"])

            # Process each sample in the batch
            for b in range(B):
                token = tokens[b]

                # Retrieve coordinate transform (Sensor -> World)
                if token in lookup.index:
                    row = lookup.loc[token]
                    w2s = np.array(row["world_to_sensor"]).reshape(4, 4)
                    s2w = np.linalg.inv(w2s)
                else:
                    # Fallback (should not happen with valid metadata)
                    s2w = np.eye(4)

                # Extract batch items
                b_scores = scores[b]
                b_clses = clses[b]
                b_reg = reg[b]
                b_wh = wh[b]
                b_rot = rot[b]
                b_z = z[b]
                b_iou = iou[b]

                # 1. Rectify Score using IoU Prediction
                # Formula: Score = score^(1-alpha) * iou^alpha
                # b_iou shape is (K, 1), squeeze to (K,)
                rect_scores = torch.pow(b_scores, 1 - alpha) * torch.pow(
                    b_iou.squeeze(-1), alpha
                )

                # 2. Filter by rectified score
                mask = rect_scores > conf_threshold
                if mask.sum() == 0:
                    results.append({"Id": token, "PredictionString": ""})
                    continue

                # Apply mask to all attributes
                f_scores = rect_scores[mask]
                f_clses = b_clses[mask]
                f_reg = b_reg[mask]
                f_wh = b_wh[mask]
                f_rot = b_rot[mask]
                f_z = b_z[mask]
                f_inds = inds[b][mask]

                # 3. Recover Box Parameters in Sensor Frame
                # Grid coordinates
                ys = (f_inds // W).float()
                xs = (f_inds % W).float()

                # Center (x, y, z) in meters
                cx = (xs + f_reg[:, 0]) * voxel_x + pc_range_x
                cy = (ys + f_reg[:, 1]) * voxel_y + pc_range_y
                cz = f_z[:, 0]

                # Dimensions (exp of log dimensions)
                cw = torch.exp(f_wh[:, 0])
                cl = torch.exp(f_wh[:, 1])
                ch = torch.exp(f_wh[:, 2])

                # Yaw (atan2 of sin, cos)
                cyaw = torch.atan2(f_rot[:, 0], f_rot[:, 1])

                # 4. Transform to Global Frame
                # Stack centers (N, 3)
                centers = torch.stack([cx, cy, cz], dim=1).cpu().numpy()

                # Apply rigid transform: P_global = P_sensor @ R^T + T
                centers_global = transform_points(
                    centers, trans=s2w[:3, 3], rot_mat=s2w[:3, :3], inverse=False
                )

                # Adjust Yaw: Global Yaw = Sensor Yaw + Yaw(Sensor->Global)
                yaw_s2w = np.arctan2(s2w[1, 0], s2w[0, 0])
                cyaw_global = cyaw.cpu().numpy() + yaw_s2w

                # 5. Format Prediction String
                pred_strs = []

                # Convert tensors to numpy for formatting
                f_scores_np = f_scores.cpu().numpy()
                f_clses_np = f_clses.cpu().numpy()
                cw_np = cw.cpu().numpy()
                cl_np = cl.cpu().numpy()
                ch_np = ch.cpu().numpy()

                for k in range(len(f_scores_np)):
                    cls_name = Config.CLASS_NAMES[f_clses_np[k]]
                    s = f_scores_np[k]
                    x, y, z = centers_global[k]
                    w, l, h = cw_np[k], cl_np[k], ch_np[k]
                    y_ang = cyaw_global[k]

                    # Format: confidence x y z w l h yaw class_name
                    pred_strs.append(
                        f"{s:.4f} {x:.4f} {y:.4f} {z:.4f} {w:.4f} {l:.4f} {h:.4f} {y_ang:.4f} {cls_name}"
                    )

                results.append({"Id": token, "PredictionString": " ".join(pred_strs)})

    # Save to CSV
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from library.config import TrainConfig, ModelConfig, DataConfig, VoxelConfig, set_seeds
from library.dataset import LidarDataset, collate_fn
from library.model import CenterPointPillars


class InferenceRunner:
    """
    Handles the inference pipeline for the CenterPoint Pillar-based detector.
    """

    def __init__(self, checkpoint_path=None):
        """
        Initialize the inference runner.
        Args:
            checkpoint_path (str, optional): Path to the model checkpoint.
                                             Defaults to TrainConfig.best_model_path.
        """
        set_seeds(TrainConfig.seed)
        self.device = torch.device(TrainConfig.device)
        self.config = TrainConfig
        self.model_config = ModelConfig
        self.voxel_config = VoxelConfig

        # Initialize Model
        self.model = CenterPointPillars().to(self.device)

        # Load Checkpoint
        ckpt_path = checkpoint_path or self.config.best_model_path
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            # print(f"Loaded model from {ckpt_path}")
        else:
            # print(f"Warning: Checkpoint {ckpt_path} not found. Using random weights.")
            pass

        self.model.eval()

    def predict_and_format(self, subset_size=None):
        """
        Runs inference on the test set, formats predictions, and saves to CSV.
        Args:
            subset_size (int, optional): If provided, only run on this many samples (for debugging).
        """
        # Initialize Test Dataset
        test_dataset = LidarDataset(
            metadata_path=DataConfig.test_metadata_path,
            split="test",
            enable_augmentation=False,
            has_targets=False,
            subset_size=subset_size,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for batch in test_loader:
                # Move inputs to device
                pillar_features = batch["pillar_features"].to(self.device)
                pillar_coords = batch["pillar_coords"].to(self.device)
                tokens = batch["tokens"]
                matrices = batch["matrices"].numpy()  # (B, 4, 4)

                batched_inputs = {
                    "pillar_features": pillar_features,
                    "pillar_coords": pillar_coords,
                    "batch_size": batch["batch_size"],
                }

                # Forward Pass
                preds = self.model(batched_inputs)

                # Decode predictions into bounding boxes (Sensor Coordinates)
                # Returns list of (N, 9) arrays
                batch_boxes = self._decode_predictions(preds)

                # Transform to Global Coordinates and Format
                for i, boxes in enumerate(batch_boxes):
                    sample_token = tokens[i]
                    matrix = matrices[i]  # Global -> Sensor

                    if len(boxes) == 0:
                        results.append({"Id": sample_token, "PredictionString": ""})
                        continue

                    # Compute Sensor -> Global transform
                    try:
                        sens_to_global = np.linalg.inv(matrix)
                    except np.linalg.LinAlgError:
                        sens_to_global = np.eye(4)

                    pred_strings = []
                    for box in boxes:
                        # Box format: [x, y, z, w, l, h, yaw, score, class_idx]
                        x, y, z, w, l, h, yaw, score, cls_idx = box

                        # 1. Transform Center
                        center_sens = np.array([x, y, z, 1.0])
                        center_glob = sens_to_global @ center_sens

                        # 2. Transform Yaw
                        # Rotate a unit vector pointing in yaw direction
                        vec_sens = np.array([np.cos(yaw), np.sin(yaw), 0.0, 0.0])
                        vec_glob = sens_to_global @ vec_sens
                        yaw_glob = np.arctan2(vec_glob[1], vec_glob[0])

                        class_name = self.model_config.class_names[int(cls_idx)]

                        # 3. Format String
                        # Task requires: score center_x center_y center_z width length height yaw class_name
                        # Note: Our box is w, l, h.
                        s = (
                            f"{score:.4f} {center_glob[0]:.4f} {center_glob[1]:.4f} {center_glob[2]:.4f} "
                            f"{w:.4f} {l:.4f} {h:.4f} {yaw_glob:.4f} {class_name}"
                        )
                        pred_strings.append(s)

                    results.append(
                        {"Id": sample_token, "PredictionString": " ".join(pred_strings)}
                    )

        # Save Submission
        df_sub = pd.DataFrame(results)
        os.makedirs(os.path.dirname(self.config.submission_path), exist_ok=True)
        df_sub.to_csv(self.config.submission_path, index=False)
        # print(f"Submission saved to {self.config.submission_path}")

    def _decode_predictions(self, preds, score_thresh=0.1, top_k=50):
        """
        Decode model outputs into bounding boxes.
        Args:
            preds (dict): Output from model (hm, center_z, dim, rot, reg).
            score_thresh (float): Minimum confidence score.
            top_k (int): Maximum number of objects per sample.
        Returns:
            list of np.ndarray: One array per sample in batch.
                                Each array: (N, 9) [x, y, z, w, l, h, yaw, score, class_idx]
        """
        hm = torch.sigmoid(preds["hm"])  # (B, C, H, W)
        center_z = preds["center_z"]  # (B, 1, H, W)
        dim = torch.exp(preds["dim"])  # (B, 3, H, W)
        rot = preds["rot"]  # (B, 2, H, W)
        reg = preds["reg"]  # (B, 2, H, W)

        batch_size, num_classes, H, W = hm.shape

        # 1. Max Pooling (NMS equivalent for CenterPoint)
        # Finds local maxima in 3x3 neighborhood
        padding = 1
        hmax = F.max_pool2d(hm, kernel_size=3, stride=1, padding=padding)
        keep = (hmax == hm).float()
        hm = hm * keep

        # 2. Top K Selection
        # Flatten: (B, C*H*W)
        hm_flat = hm.view(batch_size, -1)
        topk_scores, topk_inds = torch.topk(hm_flat, top_k)

        # Unravel indices
        topk_clses = (topk_inds // (H * W)).float()
        topk_inds = topk_inds % (H * W)
        topk_ys = (topk_inds // W).float()
        topk_xs = (topk_inds % W).float()

        # 3. Gather features at peak locations
        def gather_feat(feat, inds):
            # feat: (B, C, H, W) -> (B, H*W, C)
            feat = feat.permute(0, 2, 3, 1).contiguous()
            feat = feat.view(batch_size, -1, feat.size(3))
            # inds: (B, K) -> (B, K, C)
            inds = inds.unsqueeze(2).expand(inds.size(0), inds.size(1), feat.size(2))
            return feat.gather(1, inds)

        reg_feat = gather_feat(reg, topk_inds)  # (B, K, 2)
        z_feat = gather_feat(center_z, topk_inds)  # (B, K, 1)
        dim_feat = gather_feat(dim, topk_inds)  # (B, K, 3)
        rot_feat = gather_feat(rot, topk_inds)  # (B, K, 2)

        # 4. Decode Coordinates
        # Grid index + local offset
        xs = topk_xs + reg_feat[..., 0]
        ys = topk_ys + reg_feat[..., 1]

        # Scale to metric coordinates
        stride = self.config.out_size_factor
        voxel_size = self.voxel_config.voxel_size
        pc_range = self.voxel_config.point_cloud_range

        xs = xs * stride * voxel_size[0] + pc_range[0]
        ys = ys * stride * voxel_size[1] + pc_range[1]

        # Z coordinate
        zs = z_feat[..., 0]

        # Dimensions (w, l, h)
        ws = dim_feat[..., 0]
        ls = dim_feat[..., 1]
        hs = dim_feat[..., 2]

        # Rotation (yaw) from sin, cos
        yaws = torch.atan2(rot_feat[..., 0], rot_feat[..., 1])

        # 5. Filter and Format
        batch_results = []
        for i in range(batch_size):
            scores = topk_scores[i]
            mask = scores > score_thresh

            if mask.sum() == 0:
                batch_results.append(np.zeros((0, 9), dtype=np.float32))
                continue

            res = torch.stack(
                [
                    xs[i][mask],
                    ys[i][mask],
                    zs[i][mask],
                    ws[i][mask],
                    ls[i][mask],
                    hs[i][mask],
                    yaws[i][mask],
                    scores[mask],
                    topk_clses[i][mask],
                ],
                dim=1,
            )

            batch_results.append(res.cpu().numpy())

        return batch_results

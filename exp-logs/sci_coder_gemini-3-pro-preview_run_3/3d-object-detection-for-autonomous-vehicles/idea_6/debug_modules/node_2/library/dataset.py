import os
import numpy as np
import torch
import pandas as pd
import math
from torch.utils.data import Dataset
from library.config import Config
from library.utils import points_to_voxel, parse_label_string, box3d_to_corners
from library.data_prep import DataProcessor


def gaussian_radius(det_size, min_overlap=0.5):
    """
    Calculates the radius of the Gaussian kernel for CenterPoint targets.
    Based on the size of the object and desired overlap.
    """
    height, width = det_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2

    return min(r1, r2, r3)


def draw_gaussian(heatmap, center, radius, k=1):
    """
    Draws a Gaussian kernel on the heatmap at the specified center.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def gaussian_2d(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


class NuScenesLidarDataset(Dataset):
    def __init__(self, mode="train", subset_size=None, load_cached_data=True):
        """
        Args:
            mode: 'train', 'val', or 'test'
            subset_size: Integer to limit dataset size for debugging.
            load_cached_data: Whether to use cached metadata/GT database.
        """
        self.mode = mode
        self.config = Config
        self.processor = DataProcessor(load_cached_data=load_cached_data)

        # Load Metadata
        if mode == "train":
            self.metadata = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        elif mode == "val":
            self.metadata = pd.read_csv(self.config.VAL_METADATA_PATH)
        else:
            self.metadata = pd.read_csv(self.config.TEST_METADATA_PATH)

        if subset_size:
            self.metadata = self.metadata.iloc[:subset_size].reset_index(drop=True)

        # GT Database for Augmentation (Train only)
        self.gt_database = None
        if mode == "train" and self.config.AUG_USE_GT_SAMPLING:
            db_path = os.path.join(self.config.GT_DATABASE_DIR, "gt_database.parquet")
            if os.path.exists(db_path):
                self.gt_database = pd.read_parquet(db_path)
                # Group by class for faster sampling
                self.gt_database_grouped = {
                    k: v for k, v in self.gt_database.groupby("class_name")
                }
            else:
                print(
                    "Warning: GT Database not found. Skipping GT Sampling augmentation."
                )

        # Grid Parameters
        self.voxel_size = np.array(self.config.VOXEL_SIZE)
        self.pc_range = np.array(self.config.POINT_CLOUD_RANGE)
        self.grid_size = self.config.get_grid_size()  # [W, H, D]

        # Max Objects for Dense Heads
        self.max_objs = self.config.POST_MAX_OBJECTS

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["sample_token"]

        # 1. Load Points (Multi-sweep)
        points = self.processor.get_lidar_data(sample_token)

        # 2. Load Labels (if available)
        gt_boxes = np.zeros((0, 7))
        gt_classes = []
        if "label" in row and not pd.isna(row["label"]):
            gt_boxes, gt_classes = parse_label_string(row["label"])

        # 3. Augmentation (Train only)
        if self.mode == "train":
            points, gt_boxes, gt_classes = self._augment(points, gt_boxes, gt_classes)

        # 4. Preprocessing (Range Filter & Voxelization)
        # Filter points outside range
        mask = (
            (points[:, 0] >= self.pc_range[0])
            & (points[:, 0] < self.pc_range[3])
            & (points[:, 1] >= self.pc_range[1])
            & (points[:, 1] < self.pc_range[4])
            & (points[:, 2] >= self.pc_range[2])
            & (points[:, 2] < self.pc_range[5])
        )
        points = points[mask]

        # Voxelize
        voxels, coordinates, num_points = self._voxelize(points)

        # 5. Generate Targets (Train/Val)
        targets = {}
        if self.mode in ["train", "val"]:
            targets = self.generate_targets(gt_boxes, gt_classes)

        return {
            "voxels": voxels,  # (M, max_points, features)
            "coordinates": coordinates,  # (M, 3) [z, y, x]
            "num_points": num_points,  # (M,)
            "targets": targets,
            "sample_token": sample_token,
            "metadata": row.to_dict(),
        }

    def _augment(self, points, gt_boxes, gt_classes):
        """
        Applies GT Sampling, Rotation, Scaling, and Translation.
        """
        # A. GT Sampling (Copy-Paste)
        if self.gt_database is not None and len(gt_boxes) > 0:
            # Simple sampling strategy: try to add a few objects per class
            points, gt_boxes, gt_classes = self._gt_sampling(
                points, gt_boxes, gt_classes
            )

        # B. Global Rotation
        noise_rot = np.random.uniform(
            self.config.AUG_ROT_RANGE[0], self.config.AUG_ROT_RANGE[1]
        )
        # Rotate points
        rot_sin = np.sin(noise_rot)
        rot_cos = np.cos(noise_rot)
        rot_mat = np.array([[rot_cos, -rot_sin], [rot_sin, rot_cos]])

        points[:, :2] = points[:, :2] @ rot_mat.T

        # Rotate boxes
        if len(gt_boxes) > 0:
            gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_mat.T
            gt_boxes[:, 6] += noise_rot

        # C. Global Scaling
        noise_scale = np.random.uniform(
            self.config.AUG_SCALE_RANGE[0], self.config.AUG_SCALE_RANGE[1]
        )
        points[:, :3] *= noise_scale
        if len(gt_boxes) > 0:
            gt_boxes[:, :6] *= noise_scale

        # D. Global Translation
        noise_trans = np.random.normal(0, self.config.AUG_TRANS_STD, size=3)
        points[:, :3] += noise_trans
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] += noise_trans

        return points, gt_boxes, gt_classes

    def _gt_sampling(self, points, gt_boxes, gt_classes):
        """
        Samples objects from the database and pastes them into the scene.
        (Simplified collision checking)
        """
        new_points = [points]
        new_boxes = []
        new_classes = []

        # Existing centers for collision check
        existing_centers = gt_boxes[:, :2] if len(gt_boxes) > 0 else np.empty((0, 2))

        samples_per_class = {"car": 2, "truck": 2, "bus": 2, "pedestrian": 2}

        for cls_name, count in samples_per_class.items():
            if cls_name not in self.gt_database_grouped:
                continue

            df = self.gt_database_grouped[cls_name]
            if len(df) == 0:
                continue

            samples = df.sample(n=min(len(df), count), replace=False)

            for _, row in samples.iterrows():
                # Load points
                bin_path = os.path.join(self.config.GT_DATABASE_DIR, row["file_path"])
                if not os.path.exists(bin_path):
                    continue

                obj_points = np.fromfile(bin_path, dtype=np.float32).reshape(
                    -1, 5
                )  # Assuming 5 dims stored

                # Check collision (Naive distance check)
                # We need to place it somewhere?
                # Actually, GT database stores points relative to box center, but we need the original box location
                # to know where it belongs physically?
                # Usually GT Sampling keeps the original location of the sampled object from its source scene.
                box_k = np.array(row["box_k"])  # [x, y, z, w, l, h, yaw]

                # Check distance to existing objects
                dist = np.linalg.norm(existing_centers - box_k[:2], axis=1)
                if np.any(dist < 4.0):  # 4 meters buffer
                    continue

                # Add points (shift back to world coords)
                obj_points[:, :3] += box_k[:3]

                new_points.append(obj_points)
                new_boxes.append(box_k)
                new_classes.append(cls_name)

                # Update collision list
                existing_centers = np.vstack([existing_centers, box_k[:2]])

        if new_boxes:
            points = np.vstack(new_points)
            gt_boxes = np.vstack([gt_boxes, np.array(new_boxes)])
            gt_classes = gt_classes + new_classes

        return points, gt_boxes, gt_classes

    def _voxelize(self, points):
        """
        Converts points to pillars using numpy.
        Returns:
            voxels: (M, max_points, features)
            coordinates: (M, 3) [z, y, x]
            num_points: (M,)
        """
        if len(points) == 0:
            return (
                np.zeros((0, self.config.MAX_POINTS_PER_PILLAR, points.shape[1])),
                np.zeros((0, 3)),
                np.zeros((0,)),
            )

        # 1. Get coords
        # points_to_voxel returns (N, 3) [z, y, x] and mask
        # Note: utils.points_to_voxel returns coords for valid points only if we filter mask,
        # but here we filtered points by range already, so most should be valid.
        # However, points_to_voxel returns coords for ALL points passed.
        # Let's use the provided util carefully.

        # The util returns coords in (Z, Y, X) order for valid points.
        # It takes (N, 3) points.

        # We need to handle the fact that utils.points_to_voxel returns filtered coords.
        # We need the points corresponding to those coords.

        # Re-implementing simplified logic here to ensure mapping is correct
        voxel_size = self.voxel_size
        coors_range = self.pc_range
        grid_size = self.grid_size  # W, H, D

        coords = np.floor((points[:, :3] - coors_range[:3]) / voxel_size).astype(
            np.int32
        )

        # Filter bounds
        mask = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < grid_size[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < grid_size[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < grid_size[2])
        )

        points = points[mask]
        coords = coords[mask]

        # Coords are (x, y, z). Model expects (z, y, x) usually.
        # Let's convert to (z, y, x)
        coords = coords[:, ::-1]

        # 2. Group by unique voxel
        # Linear index for sorting: z * (H*W) + y * W + x
        # grid_size is [W, H, D]
        W, H, D = grid_size
        linear_idx = coords[:, 0] * (H * W) + coords[:, 1] * W + coords[:, 2]

        sort_idx = np.argsort(linear_idx)
        points = points[sort_idx]
        coords = coords[sort_idx]
        linear_idx = linear_idx[sort_idx]

        # Find unique change points
        unique_linear_idx, unique_indices, counts = np.unique(
            linear_idx, return_index=True, return_counts=True
        )

        # Limit number of pillars
        max_pillars = (
            self.config.MAX_PILLARS_TRAIN
            if self.mode == "train"
            else self.config.MAX_PILLARS_TEST
        )
        num_pillars = min(len(unique_linear_idx), max_pillars)

        # Select top pillars? Or random? Standard is just first N.
        # If we want to be fancy, we could shuffle, but let's stick to standard.

        voxels = np.zeros(
            (num_pillars, self.config.MAX_POINTS_PER_PILLAR, points.shape[1]),
            dtype=np.float32,
        )
        num_points = np.zeros((num_pillars,), dtype=np.int32)
        pillar_coords = np.zeros((num_pillars, 3), dtype=np.int32)

        for i in range(num_pillars):
            start = unique_indices[i]
            count = counts[i]

            # Clamp points per pillar
            num_p = min(count, self.config.MAX_POINTS_PER_PILLAR)

            voxels[i, :num_p, :] = points[start : start + num_p]
            num_points[i] = num_p
            pillar_coords[i] = coords[start]

        return voxels, pillar_coords, num_points

    def generate_targets(self, gt_boxes, gt_classes):
        """
        Generates CenterPoint targets.
        """
        # Grid dimensions (H, W) - feature map size
        # PointPillars output stride is usually 1, 2 or 4 depending on backbone.
        # Config says BACKBONE_LAYER_STRIDES = [2, 2, 2] and UPSAMPLE [1, 2, 4].
        # The final stride is effectively 1 relative to the pseudo-image if upsampled correctly,
        # OR it might be 2 or 4.
        # Let's assume stride 1 relative to the pillar grid (which is already discretized).
        # Pillar grid size: [W, H].

        W, H = self.grid_size[0], self.grid_size[1]

        # Heatmap
        hm = np.zeros((self.config.NUM_CLASSES, H, W), dtype=np.float32)

        # Regression arrays
        ind = np.zeros((self.max_objs,), dtype=np.int64)
        mask = np.zeros((self.max_objs,), dtype=np.uint8)

        # Regression targets
        # offset (2), height (1), dim (3), rot (2)
        reg_offset = np.zeros((self.max_objs, 2), dtype=np.float32)
        reg_height = np.zeros((self.max_objs, 1), dtype=np.float32)
        reg_dim = np.zeros((self.max_objs, 3), dtype=np.float32)
        reg_rot = np.zeros((self.max_objs, 2), dtype=np.float32)

        num_objs = min(len(gt_boxes), self.max_objs)

        for k in range(num_objs):
            box = gt_boxes[k]
            cls_name = gt_classes[k]
            cls_id = self.config.CLASS_TO_ID.get(cls_name, -1)

            if cls_id < 0:
                continue

            # Box: x, y, z, w, l, h, yaw
            # Convert center to grid coords
            # x_idx = (x - min_x) / voxel_x
            x, y, z = box[0], box[1], box[2]
            w, l, h = box[3], box[4], box[5]
            yaw = box[6]

            # Center in grid coords
            coor_x = (x - self.pc_range[0]) / self.voxel_size[0]
            coor_y = (y - self.pc_range[1]) / self.voxel_size[1]

            # Integer center
            ct_x = int(coor_x)
            ct_y = int(coor_y)

            # Check bounds
            if ct_x < 0 or ct_x >= W or ct_y < 0 or ct_y >= H:
                continue

            # 1. Heatmap
            # Radius based on object size in grid units
            # w, l are in meters. Convert to grid units.
            w_grid = w / self.voxel_size[0]
            l_grid = l / self.voxel_size[1]
            radius = gaussian_radius(
                (l_grid, w_grid), min_overlap=self.config.GAUSSIAN_OVERLAP
            )
            radius = max(0, int(radius))

            draw_gaussian(hm[cls_id], np.array([ct_x, ct_y]), radius)

            # 2. Regression Indices
            ind[k] = ct_y * W + ct_x
            mask[k] = 1

            # 3. Regression Targets
            # Offset: deviation from integer center
            reg_offset[k] = [coor_x - ct_x, coor_y - ct_y]

            # Height: absolute z
            reg_height[k] = [z]

            # Dim: log(l, w, h)
            # Clip to avoid log(0)
            w = max(w, 1e-4)
            l = max(l, 1e-4)
            h = max(h, 1e-4)
            reg_dim[k] = [np.log(l), np.log(w), np.log(h)]

            # Rot: sin(yaw), cos(yaw)
            reg_rot[k] = [np.sin(yaw), np.cos(yaw)]

        return {
            "heatmap": torch.from_numpy(hm),
            "inds": torch.from_numpy(ind),
            "mask": torch.from_numpy(mask),
            "offset": torch.from_numpy(reg_offset),
            "height": torch.from_numpy(reg_height),
            "dim": torch.from_numpy(reg_dim),
            "rot": torch.from_numpy(reg_rot),
        }

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function for PointPillars.
        """
        voxels_list = []
        coords_list = []
        num_points_list = []
        targets_list = []
        tokens = []
        metadata_list = []

        for i, sample in enumerate(batch):
            voxels_list.append(torch.from_numpy(sample["voxels"]))
            num_points_list.append(torch.from_numpy(sample["num_points"]))

            # Add batch index to coordinates: (batch_idx, z, y, x)
            coors = torch.from_numpy(sample["coordinates"])
            batch_idx = torch.full((coors.shape[0], 1), i, dtype=torch.int32)
            coords_list.append(torch.cat([batch_idx, coors], dim=1))

            targets_list.append(sample["targets"])
            tokens.append(sample["sample_token"])
            metadata_list.append(sample["metadata"])

        # Stack dense tensors
        voxels = torch.cat(voxels_list, dim=0)
        coordinates = torch.cat(coords_list, dim=0)
        num_points = torch.cat(num_points_list, dim=0)

        # Stack Targets (if available)
        batched_targets = {}
        if targets_list[0]:
            for key in targets_list[0].keys():
                batched_targets[key] = torch.stack(
                    [t[key] for t in targets_list], dim=0
                )

        return {
            "voxels": voxels,
            "coordinates": coordinates,
            "num_points": num_points,
            "targets": batched_targets,
            "sample_tokens": tokens,
            "metadata": metadata_list,
        }

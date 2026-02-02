import os
import numpy as np
import torch
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
from library.utils import read_points, iou_2d, limit_period


class NuScenesDataset(Dataset):
    def __init__(self, split="train", root_dir=Config.INPUT_DIR, load_cached_data=True):
        self.split = split
        self.root_dir = root_dir

        # Load Metadata
        if split == "train":
            self.metadata = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif split == "val":
            self.metadata = pd.read_csv(Config.VAL_METADATA_PATH)
        else:
            self.metadata = pd.read_csv(Config.TEST_METADATA_PATH)

        # Grid and Voxel Configuration
        self.voxel_size = np.array(Config.VOXEL_SIZE, dtype=np.float32)
        self.pc_range = np.array(Config.POINT_CLOUD_RANGE, dtype=np.float32)
        self.grid_size = np.array(Config.GRID_SIZE, dtype=np.int32)

        # Anchor Grid Configuration (Stride 2 matches typical backbone)
        self.downsample = 2
        self.feature_map_size = self.grid_size[:2] // self.downsample

        # Class Mapping
        self.class_to_id = {name: i for i, name in enumerate(Config.CLASS_NAMES)}

        # Anchor Cache
        self.cache_dir = os.path.join(Config.WORKING_DIR, "idea_1")
        os.makedirs(self.cache_dir, exist_ok=True)
        self.anchors = self._get_anchors(load_cached_data)

    def _get_anchors(self, load_cached):
        """
        Generates or loads cached anchors.
        Shape: (H, W, Num_Anchors, 7) -> Flattened to (N, 7)
        """
        cache_path = os.path.join(self.cache_dir, "anchors.npy")

        if load_cached and os.path.exists(cache_path):
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"Failed to load cached anchors: {e}. Regenerating.")

        # Generate Anchors
        x_stride = self.voxel_size[0] * self.downsample
        y_stride = self.voxel_size[1] * self.downsample

        x_offset = self.pc_range[0] + x_stride / 2
        y_offset = self.pc_range[1] + y_stride / 2

        x_centers = np.arange(self.feature_map_size[0]) * x_stride + x_offset
        y_centers = np.arange(self.feature_map_size[1]) * y_stride + y_offset

        # Meshgrid (Y, X)
        xx, yy = np.meshgrid(x_centers, y_centers)

        anchors_list = []
        for cls_name in Config.CLASS_NAMES:
            dims = Config.ANCHOR_SIZES[cls_name]  # w, l, h
            w, l, h = dims
            # Approximate sensor height offset (ground level approx -1.75m)
            z_center = -1.0

            for rot in Config.ANCHOR_ROTATIONS:
                # Create anchor for this class/rot at every grid point
                cur_anchors = np.zeros((*xx.shape, 7), dtype=np.float32)
                cur_anchors[..., 0] = xx
                cur_anchors[..., 1] = yy
                cur_anchors[..., 2] = z_center
                cur_anchors[..., 3] = w
                cur_anchors[..., 4] = l
                cur_anchors[..., 5] = h
                cur_anchors[..., 6] = rot
                anchors_list.append(cur_anchors)

        # Stack and Flatten: (H, W, Num_Anchors, 7) -> (-1, 7)
        anchors = np.stack(anchors_list, axis=-2).reshape(-1, 7)
        np.save(cache_path, anchors)
        return anchors

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        sample_token = row["sample_token"]

        # 1. Load Points
        lidar_path = os.path.join(self.root_dir, row["lidar_path"])
        points = read_points(lidar_path)  # (N, 4)

        # 2. Voxelization
        pillars, coors, n_points = self._voxelize(points)

        data_dict = {
            "pillars": pillars,
            "coors": coors,
            "n_points": n_points,
            "sample_token": sample_token,
        }

        # 3. Targets (Train/Val only)
        if self.split != "test":
            label_str = row["label"]
            gt_boxes = self._parse_labels(label_str)

            cls_map, reg_map = self._generate_targets(gt_boxes)

            data_dict["cls_map"] = cls_map
            data_dict["reg_map"] = reg_map
            # Store GT boxes for evaluation/debugging
            # Pad or pass as list in collate? We'll handle in collate if needed,
            # but usually not needed for training loop, only for val metric.

        return data_dict

    def _voxelize(self, points):
        # Filter range
        mask = (
            (points[:, 0] >= self.pc_range[0])
            & (points[:, 0] < self.pc_range[3])
            & (points[:, 1] >= self.pc_range[1])
            & (points[:, 1] < self.pc_range[4])
            & (points[:, 2] >= self.pc_range[2])
            & (points[:, 2] < self.pc_range[5])
        )
        points = points[mask]

        if len(points) == 0:
            return (
                np.zeros((1, Config.MAX_POINTS_PER_PILLAR, 9), dtype=np.float32),
                np.zeros((1, 4), dtype=np.int32),
                np.zeros((1,), dtype=np.int32),
            )

        # Calculate grid indices
        x_idx = ((points[:, 0] - self.pc_range[0]) / self.voxel_size[0]).astype(
            np.int32
        )
        y_idx = ((points[:, 1] - self.pc_range[1]) / self.voxel_size[1]).astype(
            np.int32
        )

        # Clip to ensure within bounds
        x_idx = np.clip(x_idx, 0, self.grid_size[0] - 1)
        y_idx = np.clip(y_idx, 0, self.grid_size[1] - 1)

        # Grouping key
        keys = y_idx * self.grid_size[0] + x_idx

        # Sort points by key
        sort_idx = np.argsort(keys)
        points = points[sort_idx]
        keys = keys[sort_idx]
        x_idx = x_idx[sort_idx]
        y_idx = y_idx[sort_idx]

        # Find unique pillars
        unique_keys, indices, counts = np.unique(
            keys, return_index=True, return_counts=True
        )

        # Subsample pillars if too many
        max_pillars = (
            Config.MAX_PILLARS_TRAIN
            if self.split == "train"
            else Config.MAX_PILLARS_TEST
        )
        if len(unique_keys) > max_pillars:
            choice = np.random.choice(len(unique_keys), max_pillars, replace=False)
            unique_keys = unique_keys[choice]
            indices = indices[choice]
            counts = counts[choice]

        num_pillars = len(unique_keys)

        # Allocate output buffers
        pillars = np.zeros(
            (num_pillars, Config.MAX_POINTS_PER_PILLAR, 9), dtype=np.float32
        )
        coors = np.zeros((num_pillars, 4), dtype=np.int32)  # (batch, z, y, x)
        n_points_per_pillar = np.zeros((num_pillars,), dtype=np.int32)

        # Fill pillars
        for i, (start_idx, count) in enumerate(zip(indices, counts)):
            pts = points[start_idx : start_idx + count]

            # Subsample points if too many
            if count > Config.MAX_POINTS_PER_PILLAR:
                # Deterministic slicing for speed
                pts = pts[: Config.MAX_POINTS_PER_PILLAR]
                actual_count = Config.MAX_POINTS_PER_PILLAR
            else:
                actual_count = count

            # Compute features
            # 1. Raw points (x, y, z, r)
            pillars[i, :actual_count, :4] = pts

            # 2. Offset from cluster mean (xc, yc, zc)
            cluster_mean = np.mean(pts[:, :3], axis=0)
            pillars[i, :actual_count, 4:7] = pts[:, :3] - cluster_mean

            # 3. Offset from pillar center (xp, yp)
            p_y = unique_keys[i] // self.grid_size[0]
            p_x = unique_keys[i] % self.grid_size[0]

            center_x = (
                p_x * self.voxel_size[0] + self.pc_range[0] + self.voxel_size[0] / 2
            )
            center_y = (
                p_y * self.voxel_size[1] + self.pc_range[1] + self.voxel_size[1] / 2
            )

            pillars[i, :actual_count, 7] = pts[:, 0] - center_x
            pillars[i, :actual_count, 8] = pts[:, 1] - center_y

            # Coords: (batch_idx, z, y, x)
            coors[i] = [0, 0, p_y, p_x]
            n_points_per_pillar[i] = actual_count

        return pillars, coors, n_points_per_pillar

    def _parse_labels(self, label_str):
        if pd.isna(label_str) or str(label_str).strip() == "":
            return np.zeros((0, 8), dtype=np.float32)

        parts = str(label_str).split()
        boxes = []
        stride = 8
        num_objs = len(parts) // stride

        for i in range(num_objs):
            base = i * stride
            try:
                # x, y, z, w, l, h, yaw
                params = [float(p) for p in parts[base : base + 7]]
                class_name = parts[base + 7]
                class_id = self.class_to_id.get(class_name, -1)
                if class_id != -1:
                    params.append(class_id)
                    boxes.append(params)
            except:
                continue

        if not boxes:
            return np.zeros((0, 8), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def _generate_targets(self, gt_boxes):
        num_anchors = self.anchors.shape[0]

        # Init targets
        cls_labels = np.zeros(num_anchors, dtype=np.int64)  # 0=BG
        reg_targets = np.zeros((num_anchors, 7), dtype=np.float32)

        if len(gt_boxes) == 0:
            return cls_labels, reg_targets

        # Convert to torch for IoU calculation
        anchors_t = torch.from_numpy(self.anchors)
        gt_boxes_t = torch.from_numpy(gt_boxes[:, :7])
        gt_classes = gt_boxes[:, 7].astype(np.int64)

        # Calculate IoU (x, y, w, l)
        anchors_box = anchors_t[:, [0, 1, 3, 4]]
        gt_box = gt_boxes_t[:, [0, 1, 3, 4]]

        ious = iou_2d(anchors_box, gt_box)  # (K, N)

        # Identify Anchor Classes
        # Anchor order: Class -> Rot -> Grid
        # But we flattened (H, W, Num_Types).
        # Wait, in _get_anchors: stack(..., axis=-2).
        # Shape was (H, W, Num_Types, 7).
        # Flattened to (-1, 7) means (H*W*Num_Types).
        # This implies the inner-most loop (Num_Types) is actually stride 1?
        # No, reshape(-1, 7) on (H, W, NA, 7) iterates H, then W, then NA.
        # So for a given pixel (h, w), we have NA anchors consecutively.

        num_types = len(Config.CLASS_NAMES) * len(Config.ANCHOR_ROTATIONS)
        anchor_indices = np.arange(num_anchors)
        type_idx = anchor_indices % num_types
        anchor_class_idx = type_idx // len(Config.ANCHOR_ROTATIONS)

        # Mask IoUs where classes don't match
        gt_classes_expanded = gt_classes[None, :]
        anchor_classes_expanded = anchor_class_idx[:, None]
        class_match_mask = gt_classes_expanded == anchor_classes_expanded

        ious_masked = ious.clone()
        ious_masked[~torch.from_numpy(class_match_mask)] = -1.0

        # Max IoU per anchor
        max_iou, max_idx = torch.max(ious_masked, dim=1)
        max_iou = max_iou.numpy()
        max_idx = max_idx.numpy()

        # Thresholds
        pos_thresh = np.zeros(num_anchors)
        neg_thresh = np.zeros(num_anchors)

        for i, name in enumerate(Config.CLASS_NAMES):
            p, n = Config.ANCHOR_MATCH_THRESHOLDS[name]
            mask = anchor_class_idx == i
            pos_thresh[mask] = p
            neg_thresh[mask] = n

        # Assign Labels
        # 1. Background (default 0)
        # 2. Ignore (-1)
        ignore_mask = (max_iou >= neg_thresh) & (max_iou < pos_thresh)
        cls_labels[ignore_mask] = -1

        # 3. Positives
        pos_mask = max_iou >= pos_thresh

        # 4. Force match max IoU for each GT
        gt_max_val, gt_argmax = torch.max(ious_masked, dim=0)
        gt_argmax = gt_argmax.numpy()

        for i, anchor_idx in enumerate(gt_argmax):
            if gt_max_val[i] > 0:
                pos_mask[anchor_idx] = True
                max_idx[anchor_idx] = i

        # Set positive labels (1-based class ID)
        matched_gt_classes = gt_classes[max_idx]
        cls_labels[pos_mask] = matched_gt_classes[pos_mask] + 1

        # Regression Targets
        pos_idxs = np.where(pos_mask)[0]
        if len(pos_idxs) > 0:
            a = self.anchors[pos_idxs]
            g = gt_boxes[max_idx[pos_idxs]]

            # Encode
            d_a = np.sqrt(a[:, 3] ** 2 + a[:, 4] ** 2)

            reg_targets[pos_idxs, 0] = (g[:, 0] - a[:, 0]) / d_a
            reg_targets[pos_idxs, 1] = (g[:, 1] - a[:, 1]) / d_a
            reg_targets[pos_idxs, 2] = (g[:, 2] - a[:, 2]) / a[:, 5]
            reg_targets[pos_idxs, 3] = np.log(g[:, 3] / a[:, 3])
            reg_targets[pos_idxs, 4] = np.log(g[:, 4] / a[:, 4])
            reg_targets[pos_idxs, 5] = np.log(g[:, 5] / a[:, 5])

            drot = g[:, 6] - a[:, 6]
            reg_targets[pos_idxs, 6] = limit_period(drot, 0.5, np.pi)

        return cls_labels, reg_targets

    @staticmethod
    def collate_fn(batch):
        pillars_list = []
        coors_list = []
        n_points_list = []
        cls_list = []
        reg_list = []
        tokens = []

        for i, item in enumerate(batch):
            tokens.append(item["sample_token"])
            pillars_list.append(torch.from_numpy(item["pillars"]))
            n_points_list.append(torch.from_numpy(item["n_points"]))

            coor = torch.from_numpy(item["coors"])
            coor[:, 0] = i  # Set batch index
            coors_list.append(coor)

            if "cls_map" in item:
                cls_list.append(torch.from_numpy(item["cls_map"]))
                reg_list.append(torch.from_numpy(item["reg_map"]))

        res = {
            "pillars": torch.cat(pillars_list, dim=0),
            "coors": torch.cat(coors_list, dim=0),
            "n_points": torch.cat(n_points_list, dim=0),
            "sample_tokens": tokens,
        }

        if len(cls_list) > 0:
            res["cls_targets"] = torch.stack(cls_list, dim=0)
            res["reg_targets"] = torch.stack(reg_list, dim=0)

        return res

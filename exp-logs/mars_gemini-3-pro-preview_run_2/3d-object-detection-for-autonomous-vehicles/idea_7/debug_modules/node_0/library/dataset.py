import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import math

from library.config import Config
from library.utils import box3d_to_corners, transform_points

# ==============================================================================
# Helper Functions for Heatmap Generation
# ==============================================================================


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)

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


def gaussian_radius(det_size, min_overlap=0.7):
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


# ==============================================================================
# NuScenes Dataset Class
# ==============================================================================


class NuScenesDataset(Dataset):
    def __init__(
        self,
        split,
        enable_augmentation=False,
        has_targets=False,
        load_cached_data=True,
        max_samples=None,
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            enable_augmentation (bool): Enable random flip/rot/scale.
            has_targets (bool): Generate heatmaps and regression targets.
            load_cached_data (bool): Load sweep/transform cache if available.
            max_samples (int, optional): Limit dataset size for debugging.
        """
        self.split = split
        self.enable_augmentation = enable_augmentation
        self.has_targets = has_targets
        self.max_samples = max_samples

        # Determine metadata file
        if split == "train":
            meta_file = "train_metadata.csv"
            self.data_root = os.path.join(Config.INPUT_DIR, "train_data")
        elif split == "val":
            meta_file = "val_metadata.csv"
            self.data_root = os.path.join(Config.INPUT_DIR, "train_data")
        else:
            meta_file = "test_metadata.csv"
            self.data_root = os.path.join(Config.INPUT_DIR, "test_data")

        self.metadata_path = os.path.join(Config.METADATA_DIR, meta_file)

        # Load Metadata
        print(f"Loading metadata from {self.metadata_path}...")
        self.df = pd.read_csv(self.metadata_path)

        # Parse JSON columns
        self.df["file_paths"] = self.df["file_paths"].apply(json.loads)
        self.df["annotations"] = self.df["annotations"].apply(json.loads)

        if self.max_samples is not None:
            self.df = self.df.iloc[: self.max_samples]

        # Initialize Cache for Sweeps and Transforms
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Cache filename depends on the data source (train_data vs test_data)
        cache_name = (
            "cache_train_val.parquet"
            if "train" in split or "val" in split
            else "cache_test.parquet"
        )
        self.cache_path = os.path.join(self.cache_dir, cache_name)

        self.lookup_table = self._load_or_build_cache(load_cached_data)

        # Map class names to IDs
        self.class_to_id = {name: i for i, name in enumerate(Config.CLASS_NAMES)}

        # Grid setup
        self.grid_size = Config.get_grid_size()
        self.voxel_size = np.array(Config.VOXEL_SIZE)
        self.point_cloud_range = np.array(Config.POINT_CLOUD_RANGE)
        self.down_ratio = Config.DOWN_RATIO
        self.feature_map_size = [int(x / self.down_ratio) for x in self.grid_size]

    def _load_or_build_cache(self, load_cached_data):
        """
        Loads the lookup table for multi-sweep and coordinate transforms from cache,
        or builds it from raw JSONs if cache is missing or disabled.
        """
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading dataset cache from {self.cache_path}...")
            try:
                return pd.read_parquet(self.cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Rebuilding...")

        print("Building dataset cache (this may take a few minutes)...")
        return self._build_cache()

    def _build_cache(self):
        """
        Parses sample_data.json, ego_pose.json, and calibrated_sensor.json to build
        a lookup table for:
        1. World -> Sensor transform (for GT boxes)
        2. Sweep paths and Sweep -> Current Sensor transforms (for LiDAR accumulation)
        """

        # Load raw tables
        def load_json(name):
            path = os.path.join(self.data_root, name)
            with open(path, "r") as f:
                return pd.DataFrame(json.load(f))

        df_sample_data = load_json("sample_data.json")
        df_ego_pose = load_json("ego_pose.json")
        df_calib = load_json("calibrated_sensor.json")

        # Prepare lookups
        # ego_pose: token -> {translation, rotation}
        ego_pose_map = df_ego_pose.set_index("token")[
            ["translation", "rotation"]
        ].to_dict("index")
        # calib: token -> {translation, rotation}
        calib_map = df_calib.set_index("token")[["translation", "rotation"]].to_dict(
            "index"
        )

        # Filter for LIDAR_TOP
        # We need to map sample_token -> LIDAR_TOP sample_data record
        # Note: sample_data has 'sample_token' and 'calibrated_sensor_token'
        # We first identify the calibrated_sensor_token for LIDAR_TOP
        # But easier: just filter sample_data where channel is LIDAR_TOP?
        # The raw table doesn't have 'channel' name directly, it links to calibrated_sensor.
        # But we can infer or just assume standard NuScenes structure.
        # Let's map sample_token -> list of sample_data records, then pick LIDAR_TOP.
        # Actually, let's use the file extension .bin to identify lidar records efficiently
        # or use the fact that we have file_paths in metadata.

        # Optimization: Create a map from sample_token to the specific LIDAR_TOP sample_data token
        # We iterate over our metadata df to get relevant sample_tokens
        relevant_tokens = set(self.df["token"])

        # Filter sample_data to only relevant samples and LIDAR
        # We assume LIDAR_TOP is the one we want.
        # We can find LIDAR_TOP by checking if filename ends with .bin (heuristic)
        is_lidar = df_sample_data["filename"].str.endswith(".bin")
        df_lidar = df_sample_data[is_lidar].copy()

        # We need to traverse sweeps.
        # Map: sample_data_token -> record
        sd_record_map = df_lidar.set_index("token").to_dict("index")

        # Map: sample_token -> sample_data_token (LIDAR_TOP)
        # There might be multiple lidar scans per sample (rare in NuScenes keyframes, but possible).
        # We take the one that matches the 'is_key_frame' usually, but here we just take the one linked.
        st_to_sdt = (
            df_lidar[df_lidar["sample_token"].isin(relevant_tokens)]
            .set_index("sample_token")["token"]
            .to_dict()
        )

        cache_data = []

        def get_matrix(translation, rotation):
            """Construct 4x4 homogenous transform matrix."""
            t = np.array(translation)
            q = np.array(rotation)  # w, x, y, z

            # Quaternion to Matrix
            w, x, y, z = q
            rot = np.array(
                [
                    [
                        1 - 2 * y * y - 2 * z * z,
                        2 * x * y - 2 * z * w,
                        2 * x * z + 2 * y * w,
                    ],
                    [
                        2 * x * y + 2 * z * w,
                        1 - 2 * x * x - 2 * z * z,
                        2 * y * z - 2 * x * w,
                    ],
                    [
                        2 * x * z - 2 * y * w,
                        2 * y * z + 2 * x * w,
                        1 - 2 * x * x - 2 * y * y,
                    ],
                ]
            )

            mat = np.eye(4)
            mat[:3, :3] = rot
            mat[:3, 3] = t
            return mat

        for token in self.df["token"]:
            if token not in st_to_sdt:
                # Should not happen if metadata is correct
                cache_data.append(
                    {
                        "token": token,
                        "world_to_sensor": np.eye(4).flatten().tolist(),
                        "sweep_paths": [],
                        "sweep_transforms": [],
                    }
                )
                continue

            curr_sd_token = st_to_sdt[token]
            curr_sd = sd_record_map[curr_sd_token]

            # Get Current Pose/Calib
            curr_pose = ego_pose_map[curr_sd["ego_pose_token"]]
            curr_calib = calib_map[curr_sd["calibrated_sensor_token"]]

            M_ego_to_global = get_matrix(
                curr_pose["translation"], curr_pose["rotation"]
            )
            M_sensor_to_ego = get_matrix(
                curr_calib["translation"], curr_calib["rotation"]
            )
            M_sensor_to_global = M_ego_to_global @ M_sensor_to_ego
            M_global_to_sensor = np.linalg.inv(M_sensor_to_global)

            # Sweeps
            sweep_paths = []
            sweep_transforms = []  # Transform from Sweep Sensor -> Current Sensor

            # We include the current frame as the first "sweep" (identity transform) implicitly?
            # No, usually we load current separately. The idea asks for "preceding 2 sweeps".
            # We will store paths for the *previous* sweeps.

            next_sd_token = curr_sd["prev"]
            for _ in range(Config.NUM_SWEEPS - 1):
                if not next_sd_token or next_sd_token not in sd_record_map:
                    break

                sweep_sd = sd_record_map[next_sd_token]

                # Calculate Transform Sweep -> Current
                # P_curr = T_glob_curr^-1 @ T_glob_sweep @ P_sweep
                #        = T_glob_curr^-1 @ (T_ego_glob_sweep @ T_sens_ego_sweep) @ P_sweep

                s_pose = ego_pose_map[sweep_sd["ego_pose_token"]]
                s_calib = calib_map[sweep_sd["calibrated_sensor_token"]]

                M_sweep_ego_to_global = get_matrix(
                    s_pose["translation"], s_pose["rotation"]
                )
                M_sweep_sensor_to_ego = get_matrix(
                    s_calib["translation"], s_calib["rotation"]
                )
                M_sweep_to_global = M_sweep_ego_to_global @ M_sweep_sensor_to_ego

                M_sweep_to_curr = M_global_to_sensor @ M_sweep_to_global

                # Resolve path
                # Filename is relative to input dir, e.g. "lidar/host...bin"
                # Metadata logic used "train_lidar/..."
                # The raw filename in json is usually "samples/LIDAR_TOP/..." or "sweeps/LIDAR_TOP/..."
                # We need to adjust it to match the flat structure in input/train_lidar if necessary.
                # However, the input directory structure provided in description shows "train_lidar/" containing .bin files.
                # The raw json filename might be "sweeps/LIDAR_TOP/filename.bin".
                # We just need the basename.
                basename = os.path.basename(sweep_sd["filename"])
                # Determine folder based on split
                folder = "train_lidar" if "train" in self.data_root else "test_lidar"
                full_path = os.path.join(folder, basename)

                sweep_paths.append(full_path)
                sweep_transforms.append(M_sweep_to_curr.flatten().tolist())

                next_sd_token = sweep_sd["prev"]

            cache_data.append(
                {
                    "token": token,
                    "world_to_sensor": M_global_to_sensor.flatten().tolist(),
                    "sweep_paths": sweep_paths,
                    "sweep_transforms": sweep_transforms,
                }
            )

        df_cache = pd.DataFrame(cache_data)
        df_cache.to_parquet(self.cache_path)
        return df_cache

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        token = row["token"]

        # 1. Load Cache Info
        cache_row = self.lookup_table[self.lookup_table["token"] == token].iloc[0]
        world_to_sensor = np.array(cache_row["world_to_sensor"]).reshape(4, 4)
        sweep_paths = cache_row["sweep_paths"]  # List
        sweep_transforms = cache_row["sweep_transforms"]  # List of lists

        # 2. Load Current Point Cloud
        # Path from metadata
        paths = row["file_paths"]
        # Find lidar path
        lidar_path = None
        for k, v in paths.items():
            if "LIDAR" in k:
                lidar_path = v
                break

        if lidar_path is None:
            # Fallback
            points = np.zeros((0, 4), dtype=np.float32)
        else:
            full_path = os.path.join(Config.INPUT_DIR, lidar_path)
            if os.path.exists(full_path):
                points = np.fromfile(full_path, dtype=np.float32).reshape(-1, 5)[
                    :, :4
                ]  # x,y,z,intensity
            else:
                points = np.zeros((0, 4), dtype=np.float32)

        # 3. Load and Accumulate Sweeps
        all_points = [points]
        # Append time lag indicator? Usually 0 for current, dt for others.
        # For simplicity, we just append points transformed.
        # Ideally we add a timestamp channel, but Config says PILLAR_FEATURE_DIM=64, model handles it.
        # We'll just stick to x,y,z,i.

        for sp_rel, trans_flat in zip(sweep_paths, sweep_transforms):
            sp_full = os.path.join(Config.INPUT_DIR, sp_rel)
            if os.path.exists(sp_full):
                sp_points = np.fromfile(sp_full, dtype=np.float32).reshape(-1, 5)[:, :4]

                # Transform
                M = np.array(trans_flat).reshape(4, 4)
                # Points (N, 3) -> Homogenous (N, 4) -> Transform -> (N, 3)
                xyz = sp_points[:, :3]
                xyz1 = np.hstack([xyz, np.ones((xyz.shape[0], 1))])
                xyz_t = (xyz1 @ M.T)[:, :3]

                sp_points[:, :3] = xyz_t
                all_points.append(sp_points)

        points = np.concatenate(all_points, axis=0)

        # 4. Load Annotations & Transform to Sensor Frame
        gt_boxes = []
        for ann in row["annotations"]:
            # center_x, center_y, center_z, width, length, height, yaw, class_name
            if ann["class_name"] not in self.class_to_id:
                continue

            # Box in World Frame
            # (x, y, z)
            center = np.array([ann["center_x"], ann["center_y"], ann["center_z"], 1.0])
            center_cam = (center @ world_to_sensor.T)[:3]

            # Yaw (Rotation around Z)
            # We need to rotate the orientation vector.
            # Yaw in world: angle from X-axis.
            # Global X-axis vector: (1, 0, 0)
            # Rotated vector: (cos(yaw), sin(yaw), 0)
            # Transform vector: R_global_to_sensor @ vec
            # New yaw = atan2(vec_y, vec_x)
            yaw_world = ann["yaw"]
            vec_world = np.array([np.cos(yaw_world), np.sin(yaw_world), 0.0])
            vec_sensor = vec_world @ world_to_sensor[:3, :3].T
            yaw_sensor = np.arctan2(vec_sensor[1], vec_sensor[0])

            # w, l, h
            # NuScenes: w, l, h.
            # Check Config: reg heads.

            cls_id = self.class_to_id[ann["class_name"]]

            # Box: x, y, z, w, l, h, yaw, class_id
            box = [
                center_cam[0],
                center_cam[1],
                center_cam[2],
                ann["width"],
                ann["length"],
                ann["height"],
                yaw_sensor,
                cls_id,
            ]
            gt_boxes.append(box)

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.zeros((0, 8), dtype=np.float32)
        )

        # 5. Augmentation
        if self.enable_augmentation:
            # Random Flip X
            if np.random.rand() < 0.5:
                points[:, 1] = -points[:, 1]
                if len(gt_boxes) > 0:
                    gt_boxes[:, 1] = -gt_boxes[:, 1]
                    gt_boxes[:, 6] = -gt_boxes[:, 6]  # yaw flip

            # Random Flip Y
            if np.random.rand() < 0.5:
                points[:, 0] = -points[:, 0]
                if len(gt_boxes) > 0:
                    gt_boxes[:, 0] = -gt_boxes[:, 0]
                    gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi  # yaw flip adjustment

            # Global Rotation
            noise_rotation = np.random.uniform(-np.pi / 4, np.pi / 4)
            c, s = np.cos(noise_rotation), np.sin(noise_rotation)
            rot_mat = np.array([[c, -s], [s, c]])

            points[:, :2] = points[:, :2] @ rot_mat.T
            if len(gt_boxes) > 0:
                gt_boxes[:, :2] = gt_boxes[:, :2] @ rot_mat.T
                gt_boxes[:, 6] += noise_rotation

            # Global Scaling
            scale = np.random.uniform(0.95, 1.05)
            points[:, :3] *= scale
            if len(gt_boxes) > 0:
                gt_boxes[:, :6] *= scale  # Scale x,y,z,w,l,h

        # 6. Generate Targets
        targets = {}
        if self.has_targets:
            targets = self._generate_targets(gt_boxes)

        # 7. Final Formatting
        # Shuffle points to break temporal structure if any
        np.random.shuffle(points)

        return {
            "points": torch.from_numpy(points),
            "targets": targets,
            "gt_boxes": torch.from_numpy(gt_boxes),  # For IoU loss
            "token": token,
        }

    def _generate_targets(self, gt_boxes):
        # Initialize maps
        hm = np.zeros(
            (Config.NUM_CLASSES, self.feature_map_size[1], self.feature_map_size[0]),
            dtype=np.float32,
        )
        reg = np.zeros(
            (Config.HEADS["reg"], self.feature_map_size[1], self.feature_map_size[0]),
            dtype=np.float32,
        )
        wh = np.zeros(
            (Config.HEADS["wh"], self.feature_map_size[1], self.feature_map_size[0]),
            dtype=np.float32,
        )
        rot = np.zeros(
            (Config.HEADS["rot"], self.feature_map_size[1], self.feature_map_size[0]),
            dtype=np.float32,
        )
        z_map = np.zeros(
            (Config.HEADS["z"], self.feature_map_size[1], self.feature_map_size[0]),
            dtype=np.float32,
        )

        ind = np.zeros((Config.TOP_K), dtype=np.int64)
        reg_mask = np.zeros((Config.TOP_K), dtype=np.int64)

        # We also need to return indices for gathering in loss
        # But standard CenterPoint implementation usually returns dense maps and a list of indices
        # or sparse tensors.
        # Here we will return dense maps and let the loss function handle masking via the heatmap or explicit mask.
        # To match standard implementation, we usually return:
        # { 'hm': ..., 'ind': ..., 'mask': ..., 'cat': ... }
        # But let's stick to returning dense maps for simplicity in loss if possible,
        # OR return the sparse arrays (ind, mask) which is more memory efficient for loss calc.

        # Let's populate the sparse arrays for the loss
        # Max objects = TOP_K (or more? usually 500 for training, but let's use TOP_K for simplicity)
        max_objs = Config.TOP_K

        draw_gaussian = draw_umich_gaussian

        # Filter boxes outside range
        mask = (
            (gt_boxes[:, 0] >= self.point_cloud_range[0])
            & (gt_boxes[:, 0] <= self.point_cloud_range[3])
            & (gt_boxes[:, 1] >= self.point_cloud_range[1])
            & (gt_boxes[:, 1] <= self.point_cloud_range[4])
        )
        valid_boxes = gt_boxes[mask]

        # Sort by distance? Not strictly necessary for heatmap

        num_objs = min(len(valid_boxes), max_objs)

        # Re-initialize sparse targets
        # We need arrays to gather predictions at GT locations
        target_ind = np.zeros((max_objs), dtype=np.int64)
        target_mask = np.zeros((max_objs), dtype=np.float32)
        target_cat = np.zeros((max_objs), dtype=np.int64)

        # Regression targets at indices
        # We can either return full dense regression maps (sparse populated)
        # OR return the values at the indices.
        # Standard CenterNet returns values at indices.
        target_reg = np.zeros((max_objs, 2), dtype=np.float32)
        target_wh = np.zeros((max_objs, 3), dtype=np.float32)
        target_rot = np.zeros((max_objs, 2), dtype=np.float32)
        target_z = np.zeros((max_objs, 1), dtype=np.float32)

        for k in range(num_objs):
            box = valid_boxes[k]
            cls_id = int(box[7])

            # Project to feature map
            # x, y in meters -> grid coords
            coor_x = (
                (box[0] - self.point_cloud_range[0])
                / self.voxel_size[0]
                / self.down_ratio
            )
            coor_y = (
                (box[1] - self.point_cloud_range[1])
                / self.voxel_size[1]
                / self.down_ratio
            )

            ct = np.array([coor_x, coor_y], dtype=np.float32)
            ct_int = ct.astype(np.int32)

            if not (
                0 <= ct_int[0] < self.feature_map_size[0]
                and 0 <= ct_int[1] < self.feature_map_size[1]
            ):
                continue

            # Gaussian Radius
            # Use box dimensions in BEV (w, l)
            # w, l are box[3], box[4]
            # Convert to grid scale
            w = box[3] / self.voxel_size[0] / self.down_ratio
            l = box[4] / self.voxel_size[1] / self.down_ratio
            radius = gaussian_radius((l, w), min_overlap=0.7)
            radius = max(0, int(radius))

            draw_gaussian(hm[cls_id], ct_int, radius)

            target_ind[k] = ct_int[1] * self.feature_map_size[0] + ct_int[0]
            target_mask[k] = 1
            target_cat[k] = cls_id

            # Regression Targets
            # Offset
            target_reg[k] = ct - ct_int

            # Dimensions (log)
            target_wh[k] = np.log(box[3:6])  # w, l, h

            # Rotation (sin, cos)
            target_rot[k] = [np.sin(box[6]), np.cos(box[6])]

            # Height (z)
            target_z[k] = box[2]

        return {
            "hm": torch.from_numpy(hm),
            "ind": torch.from_numpy(target_ind),
            "mask": torch.from_numpy(target_mask),
            "cat": torch.from_numpy(target_cat),
            "reg": torch.from_numpy(target_reg),
            "wh": torch.from_numpy(target_wh),
            "rot": torch.from_numpy(target_rot),
            "z": torch.from_numpy(target_z),
        }

    @staticmethod
    def collate_fn(batch):
        targets = {}

        # Stack targets if they exist
        if batch[0]["targets"]:
            for key in batch[0]["targets"]:
                targets[key] = torch.stack([b["targets"][key] for b in batch])

        # Points are variable length, keep as list
        points = [b["points"] for b in batch]

        # GT Boxes for IoU
        gt_boxes = [b["gt_boxes"] for b in batch]

        tokens = [b["token"] for b in batch]

        return {
            "points": points,
            "targets": targets,
            "gt_boxes": gt_boxes,
            "metadata": {"tokens": tokens},
        }

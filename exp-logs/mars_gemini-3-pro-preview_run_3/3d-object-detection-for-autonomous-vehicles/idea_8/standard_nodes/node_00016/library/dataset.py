import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.utils import load_point_cloud, Voxelizer


class LidarDataset(Dataset):
    def __init__(self, split="train", load_cached_data=True):
        self.split = split
        self.is_train = split == "train"
        self.voxelizer = Voxelizer()

        # Select Metadata and JSON directories
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA
            self.json_dir = Config.TRAIN_DATA_JSON
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA
            self.json_dir = Config.TRAIN_DATA_JSON
        else:  # test
            self.metadata_path = Config.TEST_METADATA
            self.json_dir = Config.TEST_DATA_JSON

        # Load Metadata
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.metadata = pd.read_csv(self.metadata_path)

        # Cache directory
        self.cache_dir = Config.CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load or Compute Transforms (World <-> Lidar)
        self.transforms = self._load_transforms(load_cached_data)

    def _load_transforms(self, load_cached):
        """
        Loads or computes the World-to-Lidar transformation matrix for each sample.
        Returns a dict: sample_token -> 4x4 homogenous matrix
        """
        cache_file = os.path.join(self.cache_dir, f"transforms_{self.split}.npy")

        if load_cached and os.path.exists(cache_file):
            try:
                print(f"Loading cached transforms from {cache_file}")
                return np.load(cache_file, allow_pickle=True).item()
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        print(f"Computing transforms for {self.split} set...")

        # Load necessary JSON tables
        def load_json(name):
            with open(os.path.join(self.json_dir, name), "r") as f:
                return json.load(f)

        sample_json = load_json("sample.json")
        sample_data_json = load_json("sample_data.json")
        ego_pose_json = load_json("ego_pose.json")
        calib_sensor_json = load_json("calibrated_sensor.json")

        # Indexing for fast lookup
        # We need sample_token -> LIDAR_TOP token
        # sample_data: token -> record
        # ego_pose: token -> record
        # calib: token -> record

        sample_map = {s["token"]: s for s in sample_json}
        ego_pose_map = {ep["token"]: ep for ep in ego_pose_json}
        calib_map = {cs["token"]: cs for cs in calib_sensor_json}

        # Map sample_token -> LIDAR sample_data record
        sample_to_lidar_sd = {}
        for sd in sample_data_json:
            if sd["filename"].endswith(".bin") and sd["sample_token"]:
                sample_to_lidar_sd[sd["sample_token"]] = sd

        transforms = {}

        for _, row in self.metadata.iterrows():
            token = row["sample_token"]

            # 1. Get sample record
            if token not in sample_map:
                continue  # Should not happen

            # 2. Get LIDAR_TOP data record
            if token not in sample_to_lidar_sd:
                continue
            sd_rec = sample_to_lidar_sd[token]

            # 3. Get Ego Pose and Calibrated Sensor
            ego_rec = ego_pose_map[sd_rec["ego_pose_token"]]
            cs_rec = calib_map[sd_rec["calibrated_sensor_token"]]

            # 4. Compute Matrices
            # Global -> Ego
            # T_ego_global = Translation * Rotation
            # We need Global -> Ego = (T_ego_global)^-1

            # Rotation (Quaternion to Matrix)
            def get_matrix(rec):
                q = rec["rotation"]
                t = rec["translation"]

                # Quaternion w, x, y, z
                w, x, y, z = q

                # Rotation matrix
                R = np.array(
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

                T = np.eye(4)
                T[:3, :3] = R
                T[:3, 3] = t
                return T

            M_ego_to_global = get_matrix(ego_rec)
            M_lidar_to_ego = get_matrix(cs_rec)

            M_lidar_to_global = M_ego_to_global @ M_lidar_to_ego
            M_global_to_lidar = np.linalg.inv(M_lidar_to_global)

            transforms[token] = M_global_to_lidar.astype(np.float32)

        # Save cache
        np.save(cache_file, transforms)
        print("Transforms computed and cached.")
        return transforms

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        token = row["sample_token"]
        lidar_path = row["lidar_path"]

        # 1. Load Points
        points = load_point_cloud(lidar_path)  # (N, 4)

        # 2. Get Transform
        # If token missing (rare edge case), use Identity
        trans_matrix = self.transforms.get(token, np.eye(4, dtype=np.float32))

        # 3. Parse Labels (if available)
        gt_boxes = np.zeros((0, 8), dtype=np.float32)
        if "label" in row and pd.notna(row["label"]):
            gt_boxes = self._parse_labels(row["label"], trans_matrix)

        # 4. Augmentation (Train only)
        if self.is_train:
            points, gt_boxes = self._augment(points, gt_boxes)

        # 5. Voxelization
        # Convert to tensor for voxelizer
        points_tensor = torch.from_numpy(points)
        voxels, coords, num_points = self.voxelizer(
            points_tensor, training=self.is_train
        )

        # If voxelizer returns None (empty cloud), handle gracefully
        if voxels is None:
            voxels = torch.zeros((1, self.voxelizer.max_points, 9))
            coords = torch.zeros((1, 3), dtype=torch.int)
            num_points = torch.zeros((1,), dtype=torch.long)

        data_dict = {
            "voxels": voxels,
            "coords": coords,
            "num_points": num_points,
            "gt_boxes": gt_boxes,
            "sample_token": token,
            "trans_matrix": trans_matrix,  # Needed for test set inversion
        }

        return data_dict

    def _parse_labels(self, label_str, trans_matrix):
        """
        Parses label string and transforms boxes to Lidar frame.
        Input: "cx cy cz w l h yaw class ..." (World Frame)
        Output: (N, 8) [x, y, z, w, l, h, yaw, class_idx] (Lidar Frame)
        """
        parts = str(label_str).strip().split()
        stride = 8
        num_objs = len(parts) // stride

        boxes = []
        for i in range(num_objs):
            off = i * stride
            try:
                cx, cy, cz = (
                    float(parts[off]),
                    float(parts[off + 1]),
                    float(parts[off + 2]),
                )
                w, l, h = (
                    float(parts[off + 3]),
                    float(parts[off + 4]),
                    float(parts[off + 5]),
                )
                yaw = float(parts[off + 6])
                class_name = parts[off + 7]

                if class_name not in Config.CLASS_MAP:
                    continue
                class_idx = Config.CLASS_MAP[class_name]

                # Transform Center
                center_h = np.array([cx, cy, cz, 1.0])
                center_lidar = (trans_matrix @ center_h)[:3]

                # Transform Yaw
                # Rotate a unit vector pointing in yaw direction
                v_yaw = np.array(
                    [np.cos(yaw), np.sin(yaw), 0.0, 0.0]
                )  # Direction vector
                v_lidar = (trans_matrix @ v_yaw)[:3]
                yaw_lidar = np.arctan2(v_lidar[1], v_lidar[0])

                boxes.append(
                    [
                        center_lidar[0],
                        center_lidar[1],
                        center_lidar[2],
                        w,
                        l,
                        h,
                        yaw_lidar,
                        class_idx,
                    ]
                )
            except ValueError:
                continue

        if len(boxes) == 0:
            return np.zeros((0, 8), dtype=np.float32)

        return np.array(boxes, dtype=np.float32)

    def _augment(self, points, boxes):
        """
        Applies random augmentation to points and boxes.
        """
        # 1. Random Flip
        if np.random.rand() < 0.5:  # Flip X
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]

        if np.random.rand() < 0.5:  # Flip Y
            points[:, 0] = -points[:, 0]
            if len(boxes) > 0:
                boxes[:, 0] = -boxes[:, 0]
                boxes[:, 6] = -(boxes[:, 6] + np.pi)

        # 2. Global Rotation
        rot_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # Rotate points
        points[:, :2] = points[:, :2] @ rot_mat.T

        # Rotate boxes
        if len(boxes) > 0:
            boxes[:, :2] = boxes[:, :2] @ rot_mat.T
            boxes[:, 6] += rot_angle

        # 3. Global Scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :6] *= scale  # Scale x,y,z,w,l,h

        return points, boxes

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable number of voxels.
        """
        voxels_list = []
        coords_list = []
        num_points_list = []
        gt_boxes_list = []
        tokens_list = []
        trans_list = []

        for i, sample in enumerate(batch):
            voxels_list.append(sample["voxels"])
            num_points_list.append(sample["num_points"])
            gt_boxes_list.append(torch.from_numpy(sample["gt_boxes"]))
            tokens_list.append(sample["sample_token"])
            trans_list.append(torch.from_numpy(sample["trans_matrix"]))

            # Coords: (M, 3) -> (z, y, x)
            # Prepend batch index -> (batch_idx, z, y, x)
            c = sample["coords"]
            b_idx = torch.full((c.shape[0], 1), i, dtype=torch.int)
            coords_list.append(torch.cat([b_idx, c], dim=1))

        return {
            "voxels": torch.cat(voxels_list, dim=0),
            "coordinates": torch.cat(coords_list, dim=0),
            "num_points": torch.cat(num_points_list, dim=0),
            "gt_boxes": gt_boxes_list,  # List of tensors (variable length)
            "sample_tokens": tokens_list,
            "trans_matrices": torch.stack(trans_list),
        }

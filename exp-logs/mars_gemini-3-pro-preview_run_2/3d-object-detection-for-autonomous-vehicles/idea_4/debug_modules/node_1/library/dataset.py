import os
import json
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config
import library.utils as utils


class Mono3DDataset(Dataset):
    def __init__(
        self, split="train", load_cached_data=True, debug=False, subset_ratio=0.1
    ):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to load data from cache if available.
            debug (bool): If True, use a subset of the data.
            subset_ratio (float): Ratio of data to use in debug mode.
        """
        self.split = split
        self.is_train = split == "train"
        self.max_objs = Config.MAX_OBJS

        # Determine paths based on split
        if split == "train":
            self.metadata_path = Config.TRAIN_METADATA
            self.data_dir = Config.TRAIN_DATA_DIR
        elif split == "val":
            self.metadata_path = Config.VAL_METADATA
            # Validation set is a subset of the training data folder
            self.data_dir = Config.TRAIN_DATA_DIR
        else:
            self.metadata_path = Config.TEST_METADATA
            self.data_dir = Config.TEST_DATA_DIR

        # Ensure cache directory exists
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        self.cache_path = os.path.join(Config.CACHE_DIR, f"{split}_data_cache.pt")

        # Load and process data
        self.samples = self._load_data(load_cached_data)

        # Debug subsetting
        if debug:
            limit = int(len(self.samples) * subset_ratio)
            self.samples = self.samples[:limit]
            print(
                f"Debug mode: Reduced {split} dataset to {len(self.samples)} samples."
            )

        print(f"Initialized Mono3DDataset ({split}) with {len(self.samples)} samples.")

    def _load_data(self, load_cached_data):
        # 1. Try loading from cache
        if load_cached_data and os.path.exists(self.cache_path):
            print(f"Loading cached {self.split} data from {self.cache_path}")
            try:
                return torch.load(self.cache_path)
            except Exception as e:
                print(f"Failed to load cache: {e}. Recomputing...")

        # 2. Compute from scratch
        print(f"Processing metadata from {self.metadata_path}...")

        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        # Load Metadata CSV
        df = pd.read_csv(self.metadata_path)

        # Parse JSON columns
        df["file_paths"] = df["file_paths"].apply(json.loads)
        df["annotations"] = df["annotations"].apply(json.loads)

        print("Loading raw JSON tables for calibration association...")
        # Load raw tables to link images to calibration
        with open(os.path.join(self.data_dir, "sample_data.json"), "r") as f:
            sample_data_df = pd.DataFrame(json.load(f))
        with open(os.path.join(self.data_dir, "calibrated_sensor.json"), "r") as f:
            calib_sensor_df = pd.DataFrame(json.load(f))
        with open(os.path.join(self.data_dir, "ego_pose.json"), "r") as f:
            ego_pose_df = pd.DataFrame(json.load(f))

        # Drop duplicates to ensure unique index for to_dict
        calib_sensor_df.drop_duplicates(subset="token", inplace=True)
        ego_pose_df.drop_duplicates(subset="token", inplace=True)

        # Create lookups for O(1) access
        calib_map = calib_sensor_df.set_index("token").to_dict("index")
        ego_map = ego_pose_df.set_index("token").to_dict("index")

        # Group sample_data by sample_token for faster filtering
        sd_grouped = sample_data_df.groupby("sample_token")

        processed_samples = []

        # Iterate over metadata samples
        for _, row in df.iterrows():
            token = row["token"]
            file_paths = row["file_paths"]
            anns = row["annotations"]

            if token not in sd_grouped.groups:
                continue

            # Get all sensor records for this sample token
            records = sd_grouped.get_group(token)

            # For each camera image in this sample
            for channel, rel_path in file_paths.items():
                # We only process camera data
                if "CAM" not in channel:
                    continue

                # Find the corresponding sample_data record
                # Match by filename basename (robust against directory changes)
                meta_fname = os.path.basename(rel_path)

                # Filter records that match this filename
                # (Assuming filenames are unique within a sample's sensors)
                match = None
                for _, sd_row in records.iterrows():
                    if os.path.basename(sd_row["filename"]) == meta_fname:
                        match = sd_row
                        break

                if match is None:
                    continue

                # Retrieve Calibration Data
                cs_token = match["calibrated_sensor_token"]
                ep_token = match["ego_pose_token"]

                if cs_token not in calib_map or ep_token not in ego_map:
                    continue

                calib = calib_map[cs_token]
                ego = ego_map[ep_token]

                sample_info = {
                    "token": token,
                    "channel": channel,
                    "image_path": os.path.join(Config.INPUT_DIR, rel_path),
                    "intrinsics": np.array(calib["camera_intrinsic"], dtype=np.float32),
                    "sensor_translation": np.array(
                        calib["translation"], dtype=np.float32
                    ),
                    "sensor_rotation": np.array(
                        calib["rotation"], dtype=np.float32
                    ),  # quaternion [w, x, y, z]
                    "ego_translation": np.array(ego["translation"], dtype=np.float32),
                    "ego_rotation": np.array(
                        ego["rotation"], dtype=np.float32
                    ),  # quaternion [w, x, y, z]
                    "annotations": anns,
                }
                processed_samples.append(sample_info)

        # Save to cache
        print(f"Saving processed data to {self.cache_path}...")
        torch.save(processed_samples, self.cache_path)

        return processed_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 1. Load and Preprocess Image
        img_path = sample["image_path"]
        img = cv2.imread(img_path)

        # Handle missing images gracefully
        if img is None:
            img = np.zeros((Config.INPUT_HEIGHT, Config.INPUT_WIDTH, 3), dtype=np.uint8)

        original_h, original_w = img.shape[:2]

        # Resize to network input size
        inp_h, inp_w = Config.INPUT_HEIGHT, Config.INPUT_WIDTH
        img_resized = cv2.resize(img, (inp_w, inp_h))

        # Normalize (HWC -> CHW, 0-1, Mean/Std)
        img_tensor = img_resized.astype(np.float32) / 255.0
        img_tensor = (img_tensor - Config.MEAN) / Config.STD
        img_tensor = img_tensor.transpose(2, 0, 1)
        img_tensor = torch.from_numpy(img_tensor).float()

        # 2. Prepare Targets
        output_h = Config.OUTPUT_HEIGHT
        output_w = Config.OUTPUT_WIDTH

        # Initialize targets
        hm = np.zeros((Config.NUM_CLASSES, output_h, output_w), dtype=np.float32)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        dim = np.zeros((self.max_objs, 3), dtype=np.float32)  # width, length, height
        depth = np.zeros((self.max_objs, 1), dtype=np.float32)
        rot = np.zeros((self.max_objs, 2), dtype=np.float32)  # sin, cos of local yaw
        offset = np.zeros((self.max_objs, 2), dtype=np.float32)

        # Scaling factors for projection
        sx = inp_w / original_w
        sy = inp_h / original_h

        num_objs = 0

        # Compute Camera Yaw in Global Frame (for local yaw calculation)
        # R_cam_global = R_ego * R_sensor
        R_ego = utils.quaternion_to_matrix(sample["ego_rotation"])
        R_sensor = utils.quaternion_to_matrix(sample["sensor_rotation"])
        R_cam_global = R_ego @ R_sensor
        # Camera Z-axis in global frame is the 3rd column
        view_dir = R_cam_global[:, 2]
        cam_yaw = np.arctan2(view_dir[1], view_dir[0])

        if self.is_train and "annotations" in sample:
            for obj in sample["annotations"]:
                if num_objs >= self.max_objs:
                    break

                cls_name = obj["class_name"]
                if cls_name not in Config.CLASS_MAP:
                    continue
                cls_id = Config.CLASS_MAP[cls_name]

                # Transform Global Center to Camera Frame
                center_global = np.array(
                    [obj["center_x"], obj["center_y"], obj["center_z"]]
                )
                center_cam = utils.global_to_camera(
                    center_global.reshape(1, 3),
                    sample["ego_translation"],
                    sample["ego_rotation"],
                    sample["sensor_translation"],
                    sample["sensor_rotation"],
                ).reshape(3)

                # Filter: Must be in front of camera
                if center_cam[2] <= 0:
                    continue

                # Project to 2D
                pts_2d, _ = utils.project_3d_to_2d(center_cam, sample["intrinsics"])
                pt_2d = pts_2d[0]

                # Apply resize scaling
                u = pt_2d[0] * sx
                v = pt_2d[1] * sy

                # Filter: Must be within image bounds
                if not (0 <= u < inp_w and 0 <= v < inp_h):
                    continue

                # Downsample to output grid
                u_out = u / Config.DOWN_RATIO
                v_out = v / Config.DOWN_RATIO
                u_int = int(u_out)
                v_int = int(v_out)

                if not (0 <= u_int < output_w and 0 <= v_int < output_h):
                    continue

                # --- Generate Targets ---

                # 1. Heatmap (Gaussian Radius based on approximate 2D size)
                # Approx 2D size using similar triangles: size_2d = size_3d * focal / depth
                # We use the max dimension for a safe radius
                size_3d = max(obj["width"], obj["length"], obj["height"])
                focal_len = (
                    sample["intrinsics"][0, 0] + sample["intrinsics"][1, 1]
                ) / 2.0
                size_2d_pixel = (size_3d * focal_len) / center_cam[2]
                size_2d_pixel *= sx  # Scale to input resolution
                size_2d_output = size_2d_pixel / Config.DOWN_RATIO

                radius = utils.gaussian_radius(
                    (size_2d_output, size_2d_output), min_overlap=0.7
                )
                radius = max(0, int(radius))

                utils.draw_gaussian(hm[cls_id], (u_int, v_int), radius)

                # 2. Indices and Mask
                ind[num_objs] = v_int * output_w + u_int
                reg_mask[num_objs] = 1

                # 3. Dimensions (width, length, height)
                dim[num_objs] = [obj["width"], obj["length"], obj["height"]]

                # 4. Depth
                depth[num_objs] = center_cam[2]

                # 5. Orientation (Local Yaw)
                # Local Yaw = Global Yaw - Camera Yaw
                local_yaw = obj["yaw"] - cam_yaw
                rot[num_objs] = [np.sin(local_yaw), np.cos(local_yaw)]

                # 6. Offset (Discretization Error)
                offset[num_objs] = [u_out - u_int, v_out - v_int]

                num_objs += 1

        # Convert targets to tensors
        targets = {
            "hm": torch.from_numpy(hm),
            "reg_mask": torch.from_numpy(reg_mask).bool(),
            "ind": torch.from_numpy(ind),
            "dim": torch.from_numpy(dim),
            "depth": torch.from_numpy(depth),
            "rot": torch.from_numpy(rot),
            "offset": torch.from_numpy(offset),
        }

        # Info dictionary for inference/decoding
        info = {
            "token": sample["token"],
            "image_path": sample["image_path"],
            "intrinsics": sample["intrinsics"],
            "ego_translation": sample["ego_translation"],
            "ego_rotation": sample["ego_rotation"],
            "sensor_translation": sample["sensor_translation"],
            "sensor_rotation": sample["sensor_rotation"],
            "original_size": np.array([original_w, original_h]),
            "cam_yaw": cam_yaw,
        }

        return img_tensor, targets, info

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.geometry import DatasetPreprocessor


class NuScenesDataset(Dataset):
    def __init__(self, metadata_path, split, config=None, load_cached_data=True):
        """
        Args:
            metadata_path (str): Path to the metadata CSV file.
            split (str): 'train', 'val', or 'test'.
            config (Config): Configuration object.
            load_cached_data (bool): Whether to use cached preprocessed data.
        """
        self.split = split
        self.config = config if config is not None else Config
        self.is_train = split == "train"

        # Ensure working directory for cache exists
        cache_dir = os.path.join(self.config.WORKING_DIR, "idea_1")
        os.makedirs(cache_dir, exist_ok=True)

        cache_file = os.path.join(cache_dir, f"cache_{split}.parquet")

        # Initialize Preprocessor from library
        preprocessor = DatasetPreprocessor(
            data_dir=(
                self.config.TRAIN_DATA_DIR
                if split != "test"
                else self.config.TEST_DATA_DIR
            )
        )

        # Load and process data (Global -> Sensor transform handled here)
        # This returns a DataFrame with columns: token, lidar_path, boxes, class_names
        self.data = preprocessor.load_and_process(
            metadata_path, cache_file, load_cached_data=load_cached_data
        )

        # Create Class Name to ID mapping
        self.class_map = {name: i + 1 for i, name in enumerate(self.config.CLASS_NAMES)}
        # 0 is reserved for background

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        token = row["token"]
        lidar_path = row["lidar_path"]

        # 1. Load Point Cloud
        full_lidar_path = os.path.join(self.config.INPUT_DIR, lidar_path)
        points = self._load_points(full_lidar_path)

        # 2. Get Annotations (if available)
        boxes = (
            np.array(row["boxes"])
            if "boxes" in row and row["boxes"] is not None
            else np.zeros((0, 7))
        )
        class_names = (
            row["class_names"]
            if "class_names" in row and row["class_names"] is not None
            else []
        )

        # Map class names to labels
        labels = np.array(
            [self.class_map.get(name, -1) for name in class_names], dtype=np.int64
        )

        # Filter out classes not in our config (return -1)
        # We will filter these out later

        # 3. Data Augmentation (Train only)
        if self.is_train:
            points, boxes = self._augment(points, boxes)

        # 4. Filter Points by Range
        points = self._filter_points_by_range(points)

        # 5. Filter Boxes by Range (keep if center is in range)
        if len(boxes) > 0:
            mask = self._filter_boxes_by_range(boxes)
            boxes = boxes[mask]
            labels = labels[mask]

            # Filter out unknown classes
            class_mask = labels != -1
            boxes = boxes[class_mask]
            labels = labels[class_mask]

        # 6. Format Output
        # Points: (N, 4) -> x, y, z, intensity
        # Boxes: (M, 7) -> x, y, z, w, l, h, yaw
        # Labels: (M,)

        sample = {
            "points": torch.from_numpy(points).float(),
            "token": token,
            "metadata": {"token": token, "num_points": points.shape[0]},
        }

        if self.split != "test":
            sample["boxes"] = torch.from_numpy(boxes).float()
            sample["labels"] = torch.from_numpy(labels).long()

        return sample

    def _load_points(self, path):
        try:
            # Load raw binary
            raw_data = np.fromfile(path, dtype=np.float32)

            # Reshape based on size
            if raw_data.size % 5 == 0:
                points = raw_data.reshape(-1, 5)
            elif raw_data.size % 4 == 0:
                points = raw_data.reshape(-1, 4)
            else:
                # Fallback or error, try 3
                points = raw_data.reshape(-1, 3)

            # Ensure we have at least 4 channels (x,y,z,i). If 3, pad with 0 intensity.
            if points.shape[1] == 3:
                intensity = np.zeros((points.shape[0], 1), dtype=np.float32)
                points = np.hstack((points, intensity))

            # Take first 4 channels
            return points[:, :4]

        except Exception as e:
            # Return empty if failed
            return np.zeros((0, 4), dtype=np.float32)

    def _filter_points_by_range(self, points):
        if len(points) == 0:
            return points

        x_min, y_min, z_min, x_max, y_max, z_max = self.config.PC_RANGE
        mask = (
            (points[:, 0] >= x_min)
            & (points[:, 0] <= x_max)
            & (points[:, 1] >= y_min)
            & (points[:, 1] <= y_max)
            & (points[:, 2] >= z_min)
            & (points[:, 2] <= z_max)
        )
        return points[mask]

    def _filter_boxes_by_range(self, boxes):
        # Filter based on center location
        x_min, y_min, z_min, x_max, y_max, z_max = self.config.PC_RANGE
        mask = (
            (boxes[:, 0] >= x_min)
            & (boxes[:, 0] <= x_max)
            & (boxes[:, 1] >= y_min)
            & (boxes[:, 1] <= y_max)
            & (boxes[:, 2] >= z_min)
            & (boxes[:, 2] <= z_max)
        )
        return mask

    def _augment(self, points, boxes):
        # 1. Random Flip along X-axis (around Y-axis)
        if np.random.rand() < 0.5:
            # Points: x -> x, y -> -y
            points[:, 1] = -points[:, 1]
            if len(boxes) > 0:
                boxes[:, 1] = -boxes[:, 1]
                boxes[:, 6] = -boxes[:, 6]  # yaw

        # 2. Global Rotation
        # Rotate around Z-axis
        rot_angle = np.random.uniform(-np.pi / 4, np.pi / 4)
        c, s = np.cos(rot_angle), np.sin(rot_angle)
        rot_mat = np.array([[c, -s], [s, c]])

        # Rotate points (x, y)
        points[:, :2] = points[:, :2] @ rot_mat.T

        if len(boxes) > 0:
            # Rotate box centers
            boxes[:, :2] = boxes[:, :2] @ rot_mat.T
            # Rotate box yaw
            boxes[:, 6] += rot_angle

        # 3. Global Scaling
        scale = np.random.uniform(0.95, 1.05)
        points[:, :3] *= scale
        if len(boxes) > 0:
            boxes[:, :6] *= scale  # Scale center and dims

        return points, boxes


def collate_fn(batch):
    """
    Custom collate function to handle variable size point clouds.
    Returns a list of point clouds and stacked targets.
    """
    batched_points = []
    batched_tokens = []
    batched_metadata = []

    batched_boxes = []
    batched_labels = []

    for sample in batch:
        batched_points.append(sample["points"])
        batched_tokens.append(sample["token"])
        batched_metadata.append(sample["metadata"])

        if "boxes" in sample:
            batched_boxes.append(sample["boxes"])
            batched_labels.append(sample["labels"])

    result = {
        "points": batched_points,  # List[Tensor(N, 4)]
        "tokens": batched_tokens,
        "metadata": batched_metadata,
    }

    if len(batched_boxes) > 0:
        result["boxes"] = batched_boxes  # List[Tensor(M, 7)]
        result["labels"] = batched_labels  # List[Tensor(M,)]

    return result


def create_data_loaders(config=None, load_cached_data=True):
    """
    Factory function to create DataLoaders for train, val, and test.
    """
    conf = config if config is not None else Config

    loaders = {}

    # Train Loader
    if os.path.exists(conf.TRAIN_METADATA_PATH):
        train_ds = NuScenesDataset(
            metadata_path=conf.TRAIN_METADATA_PATH,
            split="train",
            config=conf,
            load_cached_data=load_cached_data,
        )
        loaders["train"] = DataLoader(
            train_ds,
            batch_size=conf.BATCH_SIZE,
            shuffle=True,
            num_workers=conf.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=True,
        )

    # Val Loader
    if os.path.exists(conf.VAL_METADATA_PATH):
        val_ds = NuScenesDataset(
            metadata_path=conf.VAL_METADATA_PATH,
            split="val",
            config=conf,
            load_cached_data=load_cached_data,
        )
        loaders["val"] = DataLoader(
            val_ds,
            batch_size=conf.BATCH_SIZE,
            shuffle=False,
            num_workers=conf.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    # Test Loader
    if os.path.exists(conf.TEST_METADATA_PATH):
        test_ds = NuScenesDataset(
            metadata_path=conf.TEST_METADATA_PATH,
            split="test",
            config=conf,
            load_cached_data=load_cached_data,
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=conf.BATCH_SIZE,
            shuffle=False,
            num_workers=conf.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    return loaders

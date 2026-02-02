import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from library.config import Config
from library.transforms import (
    GTDatabaseSampler,
    RandomFlip,
    GlobalRotation,
    GlobalScaling,
    Compose,
)


class LidarDataset(Dataset):
    def __init__(self, split="train", root_dir=None):
        """
        Args:
            split (str): 'train', 'val', or 'test'.
            root_dir (str, optional): Overrides Config.DATA_DIR if provided.
        """
        self.split = split
        self.root_dir = root_dir if root_dir else Config.DATA_DIR
        self.metadata_dir = Config.METADATA_DIR

        # Load Metadata
        if split == "train":
            meta_file = "train_metadata.csv"
        elif split == "val":
            meta_file = "val_metadata.csv"
        elif split == "test":
            meta_file = "test_metadata.csv"
        else:
            raise ValueError(f"Unknown split: {split}")

        self.metadata_path = os.path.join(self.metadata_dir, meta_file)
        if not os.path.exists(self.metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {self.metadata_path}")

        self.df = pd.read_csv(self.metadata_path)

        # Class Mapping
        self.class_names = Config.CLASS_NAMES
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        # Setup Transforms
        self.transforms = None
        if split == "train":
            # Initialize augmentation pipeline
            # Note: GTDatabaseSampler will build the DB if it doesn't exist
            self.transforms = Compose(
                [
                    GTDatabaseSampler(load_cached_data=True),
                    RandomFlip(probability=0.5),
                    GlobalRotation(rotation_range=(-np.pi / 4, np.pi / 4)),
                    GlobalScaling(scale_range=(0.95, 1.05)),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_token = row["sample_token"]

        # 1. Load Point Cloud
        lidar_rel_path = row["lidar_path"]
        lidar_path = os.path.join(self.root_dir, lidar_rel_path)

        if not os.path.exists(lidar_path):
            # Fallback for missing files (should not happen with valid metadata)
            # Return empty points to avoid crashing
            points = np.zeros((0, 4), dtype=np.float32)
        else:
            try:
                points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)
            except ValueError:
                points = np.zeros((0, 4), dtype=np.float32)

        # 2. Load Annotations (if available)
        gt_boxes = []
        gt_names = []
        gt_labels = []

        if self.split in ["train", "val"]:
            label_str = row.get("label", "")
            if pd.notna(label_str) and label_str != "":
                parts = str(label_str).strip().split()
                # Format: x, y, z, w, l, h, yaw, name
                stride = 8
                num_objs = len(parts) // stride

                for i in range(num_objs):
                    offset = i * stride
                    try:
                        # Parse box
                        box = [float(parts[offset + j]) for j in range(7)]
                        class_name = parts[offset + 7]

                        if class_name in self.class_to_idx:
                            gt_boxes.append(box)
                            gt_names.append(class_name)
                            gt_labels.append(self.class_to_idx[class_name])
                    except ValueError:
                        continue

        gt_boxes = (
            np.array(gt_boxes, dtype=np.float32)
            if gt_boxes
            else np.zeros((0, 7), dtype=np.float32)
        )
        gt_names = np.array(gt_names) if gt_names else np.array([])
        gt_labels = (
            np.array(gt_labels, dtype=np.int64)
            if gt_labels
            else np.zeros((0,), dtype=np.int64)
        )

        # 3. Apply Transforms (Train only)
        if self.transforms is not None:
            data_dict = {"points": points, "gt_boxes": gt_boxes, "gt_names": gt_names}
            data_dict = self.transforms(data_dict)

            points = data_dict["points"]
            gt_boxes = data_dict["gt_boxes"]
            # gt_names updated in transforms, but we need to update labels indices
            # Re-map names to indices after augmentation (e.g. GT sampling adds new objects)
            new_names = data_dict["gt_names"]
            new_labels = [
                self.class_to_idx[n] for n in new_names if n in self.class_to_idx
            ]
            gt_labels = np.array(new_labels, dtype=np.int64)

        # 4. Prepare Output
        # Convert to torch tensors
        points_tensor = torch.from_numpy(points).float()
        gt_boxes_tensor = torch.from_numpy(gt_boxes).float()
        gt_labels_tensor = torch.from_numpy(gt_labels).long()

        metadata = {
            "sample_token": sample_token,
            "num_points": points.shape[0],
            "image_idx": idx,
        }

        return {
            "points": points_tensor,
            "gt_boxes": gt_boxes_tensor,
            "gt_labels": gt_labels_tensor,
            "metadata": metadata,
        }

    @staticmethod
    def collate_fn(batch):
        """
        Collate function for variable size point clouds.
        Returns a dictionary of lists.
        """
        batched_points = [item["points"] for item in batch]
        batched_gt_boxes = [item["gt_boxes"] for item in batch]
        batched_gt_labels = [item["gt_labels"] for item in batch]
        batched_metadata = [item["metadata"] for item in batch]

        return {
            "points": batched_points,
            "gt_boxes": batched_gt_boxes,
            "gt_labels": batched_gt_labels,
            "metadata": batched_metadata,
        }

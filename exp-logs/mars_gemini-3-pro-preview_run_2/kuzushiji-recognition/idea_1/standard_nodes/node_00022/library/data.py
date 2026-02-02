import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from library.config import Config, seed_everything, get_label_map
from library.utils import load_image, parse_labels, generate_target_mask

# Ensure reproducibility
seed_everything(Config.SEED)


class DetectionDataset(Dataset):
    """
    Dataset for Object Detection (Faster R-CNN).
    """

    def __init__(self, df, mode="train", load_cached_data=True):
        self.df = df.reset_index(drop=True)
        self.mode = mode

        # Load label mapping
        self.label2id, self.id2label = get_label_map(load_cached_data=load_cached_data)

        # Transform: ToTensor converts image to [0, 1] float
        # Cite solution_lesson_node_00020: Add ColorJitter augmentation for training
        if self.mode == "train":
            self.transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                ]
            )
        else:
            self.transforms = T.ToTensor()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        # Load Image
        img = load_image(file_path)
        # Convert to tensor (C, H, W), float [0,1]
        img_tensor = self.transforms(img)

        # For test mode, we just return the image and ID
        if self.mode == "test":
            return img_tensor, {"image_id": image_id}

        # Parse Targets
        labels_str = row.get("labels", "")
        parsed_boxes = parse_labels(labels_str)

        boxes = []
        labels = []
        area = []

        for box in parsed_boxes:
            char = box["char"]
            if char not in self.label2id:
                continue

            # Faster R-CNN expects [x1, y1, x2, y2]
            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            x1 = float(x)
            y1 = float(y)
            x2 = float(x + w)
            y2 = float(y + h)

            # Filter invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append([x1, y1, x2, y2])
            # Add 1 because 0 is reserved for background
            labels.append(self.label2id[char] + 1)
            area.append(w * h)

        # Convert to tensors
        if len(boxes) > 0:
            boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
            labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
            area_tensor = torch.as_tensor(area, dtype=torch.float32)
            iscrowd_tensor = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle images with no objects
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.int64)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
            iscrowd_tensor = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes_tensor
        target["labels"] = labels_tensor
        target["image_id"] = torch.tensor([idx])
        target["area"] = area_tensor
        target["iscrowd"] = iscrowd_tensor

        return img_tensor, target


def collate_fn(batch):
    """
    Collate function for variable size images.
    Returns tuple of lists.
    """
    return tuple(zip(*batch))


def get_dataloaders(stage, split="train", batch_size=None, debug=False):
    """
    Factory function to get dataloaders.
    """
    # Select Metadata File
    if split == "train":
        csv_path = Config.TRAIN_CSV
    elif split == "val":
        csv_path = Config.VAL_CSV
    elif split == "test":
        csv_path = Config.TEST_CSV
    else:
        raise ValueError(f"Unknown split: {split}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Metadata file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Debug subset
    if debug:
        df = df.head(Config.DEBUG_SAMPLE_SIZE)

    if stage == "detection":
        if batch_size is None:
            batch_size = Config.DET_BATCH_SIZE

        dataset = DetectionDataset(df, mode=split)

        shuffle = split == "train"

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )
    else:
        raise ValueError(f"Unknown stage: {stage}")

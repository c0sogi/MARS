import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from library.config import Config, seed_everything, get_label_map
from library.utils import load_image, parse_labels

# Ensure reproducibility
seed_everything(Config.SEED)


class DetectionDataset(Dataset):
    """
    Dataset for Object Detection (Faster R-CNN).
    Cite solution_lesson_node_00001: Using object detection instead of segmentation.
    """

    def __init__(self, df, mode="train", transform=None):
        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

        # Load label mapping
        self.label2id, self.id2label = get_label_map(load_cached_data=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row["image_id"]
        file_path = row["file_path"]

        # Load Image
        img = load_image(file_path)

        # Cite solution_lesson_node_00020: Photometric Augmentation
        if self.mode == "train":
            img_pil = T.ToPILImage()(img)
            # Apply ColorJitter
            jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
            img_aug = jitter(img_pil)
            img_tensor = T.functional.to_tensor(img_aug)
        else:
            # Convert to tensor (0-1 float)
            img_tensor = T.functional.to_tensor(img)

        if self.mode == "test":
            return img_tensor, image_id

        # Parse Labels
        labels_str = row.get("labels", "")
        parsed_boxes = parse_labels(labels_str)

        boxes = []
        labels = []

        for box in parsed_boxes:
            char = box["char"]
            if char not in self.label2id:
                continue

            x, y, w, h = box["x"], box["y"], box["w"], box["h"]
            # Faster R-CNN expects [x1, y1, x2, y2]
            x2 = x + w
            y2 = y + h

            # Filter invalid boxes
            if x2 <= x or y2 <= y:
                continue

            boxes.append([x, y, x2, y2])
            labels.append(self.label2id[char])

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            # Handle images with no labels
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))


def get_dataloaders(split="train", batch_size=None, debug=False):
    """
    Factory function to get dataloaders for detection.
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

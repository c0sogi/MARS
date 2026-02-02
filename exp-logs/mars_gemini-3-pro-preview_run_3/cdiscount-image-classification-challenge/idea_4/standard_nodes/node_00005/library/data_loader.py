import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import read_bson_record, extract_images_from_bson, preprocess_image


def get_label_mapping():
    """
    Reads category_names.csv and returns mappings:
    - raw_to_idx: dict mapping category_id (int) -> index (0..N-1)
    - idx_to_raw: dict mapping index (0..N-1) -> category_id (int)
    """
    df = pd.read_csv(Config.CATEGORY_NAMES)
    # Sort to ensure deterministic mapping
    unique_cats = sorted(df["category_id"].unique())
    raw_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    idx_to_raw = {i: cat for i, cat in enumerate(unique_cats)}
    return raw_to_idx, idx_to_raw


class BSONProductDataset(Dataset):
    """
    Dataset for reading raw BSON data.
    Returns:
        _id (int): Product ID
        images (Tensor): Stack of preprocessed images (N, 3, 224, 224)
        label (int): Raw category_id (or -1 if test)
    """

    def __init__(self, metadata_path, transform=None, limit=None):
        self.metadata = pd.read_csv(metadata_path)

        # Apply debug limit if configured or passed
        if limit is not None:
            self.metadata = self.metadata.iloc[:limit]
        elif Config.DEBUG:
            self.metadata = self.metadata.iloc[: Config.DEBUG_SIZE]

        self.transform = transform
        self.input_dir = Config.INPUT_DIR

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # Extract metadata fields
        _id = int(row["_id"])
        offset = int(row["bson_offset"])
        length = int(row["bson_length"])
        file_path = os.path.join(self.input_dir, row["file_path"])

        # Determine label
        label = -1
        if "category_id" in row and not pd.isna(row["category_id"]):
            label = int(row["category_id"])

        # Read BSON record
        # Note: We open/close file per record. OS page cache handles performance.
        try:
            data = read_bson_record(file_path, offset, length)
            img_bytes_list = extract_images_from_bson(data)
        except Exception as e:
            print(f"Error reading record {_id} at offset {offset}: {e}")
            img_bytes_list = []

        # Preprocess images
        processed_imgs = []
        for b in img_bytes_list:
            tensor = preprocess_image(b, self.transform)
            processed_imgs.append(tensor)

        # Handle case with no images (robustness)
        if not processed_imgs:
            # Return a single black image
            processed_imgs.append(torch.zeros(3, 224, 224))

        # Stack into (N_imgs, 3, H, W)
        img_stack = torch.stack(processed_imgs)

        return _id, img_stack, label


def collate_bson_products(batch):
    """
    Custom collate function for BSONProductDataset.
    Flattens the image stacks from multiple products into a single batch for CNN inference.

    Returns:
        dict: {
            'ids': LongTensor (Batch_Size),
            'images': FloatTensor (Total_Images, 3, H, W),
            'labels': LongTensor (Batch_Size),
            'counts': LongTensor (Batch_Size) - number of images per product
        }
    """
    ids = []
    labels = []
    image_tensors = []
    counts = []

    for _id, img_stack, label in batch:
        ids.append(_id)
        labels.append(label)
        image_tensors.append(img_stack)
        counts.append(img_stack.shape[0])

    # Concatenate all images
    batch_images = torch.cat(image_tensors, dim=0)

    return {
        "ids": torch.tensor(ids, dtype=torch.long),
        "images": batch_images,
        "labels": torch.tensor(labels, dtype=torch.long),
        "counts": torch.tensor(counts, dtype=torch.long),
    }


class EmbeddingDataset(Dataset):
    """
    Dataset for serving pre-computed embeddings.
    Features are expected to be in memory (numpy arrays).
    """

    def __init__(self, features, labels=None, ids=None):
        """
        Args:
            features (np.ndarray): (N, D) float array
            labels (np.ndarray, optional): (N,) int array (mapped indices 0..C-1)
            ids (np.ndarray, optional): (N,) int array (raw product ids)
        """
        self.features = torch.from_numpy(features).float()

        self.labels = None
        if labels is not None:
            self.labels = torch.from_numpy(labels).long()

        self.ids = None
        if ids is not None:
            self.ids = torch.from_numpy(ids).long()

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return dict for flexibility
        item = {"feature": self.features[idx]}

        if self.labels is not None:
            item["label"] = self.labels[idx]

        if self.ids is not None:
            item["_id"] = self.ids[idx]

        return item


def get_bson_loader(
    metadata_path, batch_size=128, shuffle=False, num_workers=4, limit=None
):
    """
    Factory for raw BSON data loader.
    """
    dataset = BSONProductDataset(metadata_path, limit=limit)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_bson_products,
        pin_memory=True,
    )
    return loader


def get_embedding_loader(
    features, labels=None, ids=None, batch_size=2048, shuffle=True, num_workers=4
):
    """
    Factory for embedding data loader.
    """
    dataset = EmbeddingDataset(features, labels, ids)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader

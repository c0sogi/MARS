import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from library.config import CACHE_DIR, IMG_SIZE, DEBUG_SAMPLE_SIZE
from library.utils import extract_images_from_bson, get_transforms


def load_metadata(path, load_cached_data=True, debug_size=None):
    """
    Loads metadata from CSV, using Parquet caching for speed.
    Optionally subsets the data for debugging.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_filename = os.path.basename(path).replace(".csv", ".parquet")
    if debug_size is not None:
        cache_filename = cache_filename.replace(
            ".parquet", f"_debug_{debug_size}.parquet"
        )

    cache_path = os.path.join(CACHE_DIR, cache_filename)

    if load_cached_data and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)

    # Load from source
    df = pd.read_csv(path)

    # Apply debug slicing if requested
    if debug_size is not None and len(df) > debug_size:
        df = df.iloc[:debug_size]

    # Save to cache
    if load_cached_data:
        df.to_parquet(cache_path)

    return df


def get_label_map(category_names_path):
    """
    Creates a consistent mapping between category_id and class index.
    Based on the global category_names.csv to ensure all classes are covered.
    """
    df = pd.read_csv(category_names_path)
    unique_cats = sorted(df["category_id"].unique())
    cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
    idx_to_cat = {i: cat for i, cat in enumerate(unique_cats)}
    return cat_to_idx, idx_to_cat


class ProductImageDataset(Dataset):
    """
    Dataset that reads images directly from BSON files using metadata offsets.
    Returns all images associated with a single product.
    """

    def __init__(
        self,
        metadata_path,
        bson_path,
        category_names_path,
        transform=None,
        load_cached_data=True,
        is_test=False,
        debug_size=DEBUG_SAMPLE_SIZE,
    ):
        self.metadata = load_metadata(metadata_path, load_cached_data, debug_size)
        self.bson_path = bson_path
        self.transform = transform if transform else get_transforms(IMG_SIZE)
        self.is_test = is_test
        self.file_handle = None

        # Initialize label mapping
        self.cat_to_idx, self.idx_to_cat = get_label_map(category_names_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Lazy file opening (one handle per worker process)
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        row = self.metadata.iloc[idx]
        offset = int(row["bson_offset"])
        length = int(row["bson_length"])
        product_id = int(row["_id"])

        # Seek and read
        self.file_handle.seek(offset)
        data = self.file_handle.read(length)

        # Extract images (returns list of RGB numpy arrays)
        images_np = extract_images_from_bson(data)

        # Handle rare edge case of 0 images (data integrity fallback)
        if len(images_np) == 0:
            images_np = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

        # Apply transforms
        images_tensor = [self.transform(img) for img in images_np]

        # Stack to create [K, C, H, W] tensor, where K is num images for this product
        images_tensor = torch.stack(images_tensor)

        # Get Label
        label = -1
        if not self.is_test:
            cat_id = int(row["category_id"])
            label = self.cat_to_idx.get(cat_id, -1)

        return images_tensor, label, product_id


def product_collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the image batch for the CNN but keeps track of counts for aggregation.

    Args:
        batch: List of tuples (images_tensor, label, product_id)
               images_tensor shape: [K, C, H, W]

    Returns:
        flat_images: [Total_K, C, H, W]
        counts: [Batch_Size] (number of images per product)
        labels: [Batch_Size]
        ids: [Batch_Size]
    """
    images_list = []
    labels_list = []
    ids_list = []
    counts_list = []

    for imgs, label, pid in batch:
        images_list.append(imgs)
        labels_list.append(label)
        ids_list.append(pid)
        counts_list.append(imgs.shape[0])

    # Concatenate all images along the batch dimension
    flat_images = torch.cat(images_list, dim=0)

    labels = torch.tensor(labels_list, dtype=torch.long)
    ids = torch.tensor(ids_list, dtype=torch.long)
    counts = torch.tensor(counts_list, dtype=torch.long)

    return flat_images, counts, labels, ids


class EmbeddingDataset(Dataset):
    """
    Dataset for the second stage: Training the MLP on pre-computed embeddings.
    """

    def __init__(self, embeddings, labels=None):
        """
        Args:
            embeddings (np.ndarray or torch.Tensor): Shape [N, Embedding_Dim]
            labels (np.ndarray or torch.Tensor, optional): Shape [N]
        """
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.as_tensor(labels, dtype=torch.long)
        else:
            self.labels = None

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.embeddings[idx], self.labels[idx]
        return self.embeddings[idx]

import os
import torch
import numpy as np
import pandas as pd
import timm
from torch.utils.data import Dataset, DataLoader
from torch_scatter import scatter_mean
from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    BATCH_SIZE_EXTRACT,
    NUM_WORKERS,
    DEVICE,
    SEED,
)
from library.data_utils import BSONIterator, preprocess_image, seed_everything


class BSONDataset(Dataset):
    """
    PyTorch Dataset that retrieves images from BSON files using metadata indices.
    """

    def __init__(self, metadata_path, bson_path, is_test=False):
        self.metadata = pd.read_csv(metadata_path)
        self.bson_iterator = BSONIterator(bson_path, self.metadata)
        self.is_test = is_test

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve raw image bytes from BSON
        img_bytes_list = self.bson_iterator.get_images(idx)

        # Preprocess images
        tensors = []
        for b in img_bytes_list:
            t = preprocess_image(b)
            if t is not None:
                tensors.append(t)

        # Fallback for records with no valid images (rare/corrupt)
        if not tensors:
            # Return a single zero tensor to maintain batch integrity
            tensors = [torch.zeros((3, 224, 224), dtype=torch.float32)]

        # Determine target: _id for test, category_id for train/val
        if self.is_test:
            label_or_id = self.metadata.iloc[idx]["_id"]
        else:
            label_or_id = self.metadata.iloc[idx]["category_id"]

        return tensors, label_or_id


def collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the list of image lists into a single batch tensor and creates
    an index mapping to aggregate them back to product-level features.
    """
    all_images = []
    batch_indices = []
    targets = []

    for batch_idx, (tensors, target) in enumerate(batch):
        all_images.extend(tensors)
        # Map these images to the current product's index in the batch
        batch_indices.extend([batch_idx] * len(tensors))
        targets.append(target)

    # Stack all images into a single tensor: (Total_Images, 3, H, W)
    all_images = torch.stack(all_images)

    # Indices for scatter reduction: (Total_Images,)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)

    # Targets: (Batch_Size,) - int64
    targets = torch.tensor(targets, dtype=torch.long)

    return all_images, batch_indices, targets


def _process_split(
    metadata_path, bson_path, output_feat_path, output_label_path, model, is_test=False
):
    """
    Internal function to process a single dataset split (Train, Val, or Test).
    Extracts features and saves them to disk.
    """
    print(f"Processing split: {os.path.basename(metadata_path)}...")

    # Initialize Dataset
    dataset = BSONDataset(metadata_path, bson_path, is_test=is_test)

    # Initialize DataLoader
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE_EXTRACT,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
        prefetch_factor=2 if NUM_WORKERS > 0 else None,
    )

    all_features = []
    all_labels = []

    # Inference Loop
    with torch.no_grad():
        for images, batch_indices, targets in loader:
            images = images.to(DEVICE, non_blocking=True)
            batch_indices = batch_indices.to(DEVICE, non_blocking=True)

            # Forward Pass: (Total_Images, 1280)
            # EfficientNet-B0 with num_classes=0 returns global pooled features
            features = model(images)

            # Mean Pooling per Product: (Batch_Size, 1280)
            # Aggregates features of all images belonging to the same product
            product_features = scatter_mean(features, batch_indices, dim=0)

            # Move to CPU and collect
            all_features.append(product_features.cpu().numpy())
            all_labels.append(targets.numpy())

    # Concatenate all batches
    if len(all_features) > 0:
        final_features = np.concatenate(all_features, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
    else:
        final_features = np.array([])
        final_labels = np.array([])

    print(f"  > Extracted features shape: {final_features.shape}")
    print(f"  > Extracted labels/IDs shape: {final_labels.shape}")

    # Save to disk
    np.save(output_feat_path, final_features)
    np.save(output_label_path, final_labels)

    # Verify file creation
    if not os.path.exists(output_feat_path):
        raise RuntimeError(f"Failed to save features to {output_feat_path}")


def extract_features(load_cached_data=True):
    """
    Main entry point to extract features for Train, Val, and Test sets.

    Args:
        load_cached_data (bool): If True, checks for existing .npy files and skips extraction if found.
    """
    seed_everything(SEED)

    # Ensure working directory exists
    os.makedirs(os.path.dirname(TRAIN_FEATURES_PATH), exist_ok=True)

    # Check if all files exist to potentially skip everything
    all_files_exist = (
        os.path.exists(TRAIN_FEATURES_PATH)
        and os.path.exists(TRAIN_LABELS_PATH)
        and os.path.exists(VAL_FEATURES_PATH)
        and os.path.exists(VAL_LABELS_PATH)
        and os.path.exists(TEST_FEATURES_PATH)
        and os.path.exists(TEST_IDS_PATH)
    )

    if load_cached_data and all_files_exist:
        print("All cached features found. Skipping extraction.")
        return

    print("Initializing EfficientNet-B0 for feature extraction...")
    # Load Pretrained EfficientNet-B0
    # num_classes=0 removes the head and returns the pooling layer output (1280 dim)
    model = timm.create_model("tf_efficientnet_b0", pretrained=True, num_classes=0)
    model.eval()
    model.to(DEVICE)

    # 1. Process Train
    if not (
        load_cached_data
        and os.path.exists(TRAIN_FEATURES_PATH)
        and os.path.exists(TRAIN_LABELS_PATH)
    ):
        _process_split(
            TRAIN_META_PATH,
            TRAIN_BSON_PATH,
            TRAIN_FEATURES_PATH,
            TRAIN_LABELS_PATH,
            model,
            is_test=False,
        )
    else:
        print("Train features cached.")

    # 2. Process Val
    if not (
        load_cached_data
        and os.path.exists(VAL_FEATURES_PATH)
        and os.path.exists(VAL_LABELS_PATH)
    ):
        _process_split(
            VAL_META_PATH,
            TRAIN_BSON_PATH,
            VAL_FEATURES_PATH,
            VAL_LABELS_PATH,
            model,
            is_test=False,
        )
    else:
        print("Val features cached.")

    # 3. Process Test
    if not (
        load_cached_data
        and os.path.exists(TEST_FEATURES_PATH)
        and os.path.exists(TEST_IDS_PATH)
    ):
        _process_split(
            TEST_META_PATH,
            TEST_BSON_PATH,
            TEST_FEATURES_PATH,
            TEST_IDS_PATH,
            model,
            is_test=True,
        )
    else:
        print("Test features cached.")

    print("Feature extraction complete.")

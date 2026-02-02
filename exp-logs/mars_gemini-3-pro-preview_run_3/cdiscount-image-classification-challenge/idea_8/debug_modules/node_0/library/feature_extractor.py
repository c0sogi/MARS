import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
import sys

# Import from provided libraries
from library.config import Config
from library.data_utils import BSONIterator, HierarchyMapper


class BSONDataset(Dataset):
    def __init__(self, meta_df, bson_path, hierarchy_mapper=None, is_test=False):
        self.meta_df = meta_df
        self.bson_path = bson_path
        self.mapper = hierarchy_mapper
        self.is_test = is_test
        self.iterator = None

        # Standard ImageNet normalization
        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        # Lazy initialization for worker safety
        if self.iterator is None:
            self.iterator = BSONIterator(self.bson_path)

        row = self.meta_df.iloc[idx]
        offset = int(row["bson_offset"])
        length = int(row["bson_length"])

        # Retrieve images
        # BSONIterator returns list of RGB numpy arrays resized to Config.RESIZE_SIZE
        images_list = self.iterator.get_images(offset, length)

        if not images_list:
            # Fallback for empty image list (edge case)
            tensor_imgs = torch.zeros((1, 3, Config.RESIZE_SIZE, Config.RESIZE_SIZE))
        else:
            tensors = [self.transform(img) for img in images_list]
            tensor_imgs = torch.stack(tensors)  # Shape: (Num_Images, 3, H, W)

        product_id = int(row["_id"])

        if self.is_test:
            return tensor_imgs, product_id
        else:
            cat_id = int(row["category_id"])
            l1, l2, l3 = self.mapper.get_labels(cat_id)
            # Return hierarchical labels
            labels = np.array([l1, l2, l3], dtype=np.int64)
            return tensor_imgs, labels, product_id


def collate_fn(batch):
    """
    Custom collate to handle variable number of images per product.
    Flattens images into a single batch tensor and returns counts to reconstruct.
    """
    if len(batch[0]) == 3:  # Train/Val: (imgs, labels, ids)
        imgs, labels, ids = zip(*batch)

        # Flatten list of tensors
        flat_imgs = torch.cat(imgs, dim=0)
        counts = [img.shape[0] for img in imgs]

        labels = torch.tensor(np.stack(labels), dtype=torch.long)
        ids = torch.tensor(ids, dtype=torch.long)

        return flat_imgs, counts, labels, ids
    else:  # Test: (imgs, ids)
        imgs, ids = zip(*batch)

        flat_imgs = torch.cat(imgs, dim=0)
        counts = [img.shape[0] for img in imgs]

        ids = torch.tensor(ids, dtype=torch.long)

        return flat_imgs, counts, ids


class FeatureExtractor:
    def __init__(self):
        self.device = Config.DEVICE
        self.inference_batch_size = 256  # Batch size for ResNet inference

        # Initialize Model
        print("Initializing ResNet50 backbone...")
        weights = ResNet50_Weights.IMAGENET1K_V1
        self.model = resnet50(weights=weights)
        # Remove classification head (fc) - we want the output of the pooling layer
        # ResNet forward: conv1 -> ... -> avgpool -> flatten -> fc
        # We can just set fc to Identity, but standard ResNet avgpool output is (B, 2048, 1, 1)
        # We want (B, 2048).
        self.model.fc = nn.Identity()
        self.model.to(self.device)
        self.model.eval()

    def _run_inference(self, dataloader, desc):
        """
        Runs inference on the dataloader and aggregates features.
        """
        all_features = []
        all_labels = []
        all_ids = []

        print(f"Starting feature extraction: {desc}")

        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                if len(batch_data) == 4:
                    flat_imgs, counts, labels, ids = batch_data
                    all_labels.append(labels.numpy())
                else:
                    flat_imgs, counts, ids = batch_data

                flat_imgs = flat_imgs.to(self.device)

                # Forward pass
                # Output of resnet50 with fc=Identity is (Total_Images, 2048)
                features = self.model(flat_imgs)

                # Aggregate per product (Mean Pooling)
                # Split features back into per-product groups based on 'counts'
                features_split = torch.split(features, counts)

                # Stack mean-pooled vectors
                # torch.stack([t.mean(dim=0) for t in features_split]) is efficient enough
                product_features = torch.stack([f.mean(dim=0) for f in features_split])

                all_features.append(product_features.cpu().numpy())
                all_ids.append(ids.numpy())

                if batch_idx % 100 == 0:
                    print(f"Processed batch {batch_idx}/{len(dataloader)}")

        # Concatenate all
        final_features = np.concatenate(all_features, axis=0)
        final_ids = np.concatenate(all_ids, axis=0)

        final_labels = None
        if all_labels:
            final_labels = np.concatenate(all_labels, axis=0)

        return final_features, final_labels, final_ids

    def process_and_cache(self, load_cached_data=True):
        """
        Main entry point. Checks cache, processes data if needed, saves results.
        """
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Paths
        paths = {
            "train": (Config.TRAIN_FEATURES_PATH, Config.TRAIN_LABELS_PATH, None),
            "val": (Config.VAL_FEATURES_PATH, Config.VAL_LABELS_PATH, None),
            "test": (Config.TEST_FEATURES_PATH, None, Config.TEST_IDS_PATH),
        }

        # Check if all exist
        all_exist = True
        if load_cached_data:
            if not (
                os.path.exists(Config.TRAIN_FEATURES_PATH)
                and os.path.exists(Config.TRAIN_LABELS_PATH)
            ):
                all_exist = False
            if not (
                os.path.exists(Config.VAL_FEATURES_PATH)
                and os.path.exists(Config.VAL_LABELS_PATH)
            ):
                all_exist = False
            if not (
                os.path.exists(Config.TEST_FEATURES_PATH)
                and os.path.exists(Config.TEST_IDS_PATH)
            ):
                all_exist = False
        else:
            all_exist = False

        if all_exist:
            print("All features found in cache. Skipping extraction.")
            return

        print("Cache miss or force reload. Starting extraction pipeline...")

        # Load Hierarchy Mapper
        mapper = HierarchyMapper(load_cached_data=True)

        # 1. Process Train
        print("Loading Train Metadata...")
        train_df = pd.read_csv(Config.TRAIN_META)
        if Config.DEBUG:
            print(
                f"DEBUG: Subsampling train from {len(train_df)} to {Config.DEBUG_SAMPLE_SIZE}"
            )
            train_df = train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        train_ds = BSONDataset(train_df, Config.TRAIN_BSON, mapper, is_test=False)
        train_loader = DataLoader(
            train_ds,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        feats, labels, ids = self._run_inference(train_loader, "Train")
        print(f"Saving Train features: {feats.shape}, Labels: {labels.shape}")
        np.save(Config.TRAIN_FEATURES_PATH, feats)
        np.save(Config.TRAIN_LABELS_PATH, labels)
        # We don't strictly need train IDs for training, but good for debugging if needed.

        # 2. Process Val
        print("Loading Val Metadata...")
        val_df = pd.read_csv(Config.VAL_META)
        if Config.DEBUG:
            val_df = val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        val_ds = BSONDataset(val_df, Config.TRAIN_BSON, mapper, is_test=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        feats, labels, ids = self._run_inference(val_loader, "Val")
        print(f"Saving Val features: {feats.shape}, Labels: {labels.shape}")
        np.save(Config.VAL_FEATURES_PATH, feats)
        np.save(Config.VAL_LABELS_PATH, labels)

        # 3. Process Test
        print("Loading Test Metadata...")
        test_df = pd.read_csv(Config.TEST_META)
        if Config.DEBUG:
            test_df = test_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        test_ds = BSONDataset(test_df, Config.TEST_BSON, mapper=None, is_test=True)
        test_loader = DataLoader(
            test_ds,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        feats, labels, ids = self._run_inference(test_loader, "Test")
        print(f"Saving Test features: {feats.shape}, IDs: {ids.shape}")
        np.save(Config.TEST_FEATURES_PATH, feats)
        np.save(Config.TEST_IDS_PATH, ids)

        print("Feature extraction complete.")


def run_feature_extraction(load_cached_data=True):
    extractor = FeatureExtractor()
    extractor.process_and_cache(load_cached_data=load_cached_data)

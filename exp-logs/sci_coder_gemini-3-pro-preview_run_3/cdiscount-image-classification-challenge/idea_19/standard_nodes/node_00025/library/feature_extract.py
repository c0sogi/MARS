import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from torch_scatter import scatter_mean

from library.config import Config
from library.utils import BSONImageLoader, HierarchyMapper

# ==========================================
# DATASET & TRANSFORMS
# ==========================================

# Standard ImageNet normalization
TRANSFORM = T.Compose(
    [T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
)


class BSONDataset(Dataset):
    """
    Dataset that reads images from BSON files based on metadata offsets.
    Returns:
        images_tensor: (Num_Images, 3, H, W)
        target: category_id (train/val) or _id (test)
    """

    def __init__(self, metadata_df, bson_path, transform=None, is_test=False):
        self.metadata = metadata_df
        self.bson_path = bson_path
        self.transform = transform
        self.is_test = is_test
        self.loader = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Initialize loader in the worker process
        if self.loader is None:
            self.loader = BSONImageLoader(self.bson_path)
            self.loader.open()

        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read images (returns list of numpy arrays)
        images_np = self.loader.read_images(offset, length)

        # Fallback for corrupt/empty records (though unlikely)
        if not images_np:
            images_np = [
                np.zeros((Config.IMG_SIZE, Config.IMG_SIZE, 3), dtype=np.uint8)
            ]

        # Apply transforms
        images_t = []
        for img in images_np:
            if self.transform:
                images_t.append(self.transform(img))
            else:
                images_t.append(T.ToTensor()(img))

        # Stack into (Num_Images, 3, H, W)
        images_tensor = torch.stack(images_t)

        if self.is_test:
            return images_tensor, row["_id"]
        else:
            return images_tensor, row["category_id"]


def collate_fn(batch):
    """
    Custom collate to handle variable number of images per product.
    Flattens images into a single batch and creates indices for scatter pooling.
    """
    images_list = []
    batch_indices = []
    labels = []

    for i, (imgs, label) in enumerate(batch):
        images_list.append(imgs)
        # Create indices [i, i, i...] mapping these images to product i
        batch_indices.append(torch.full((imgs.shape[0],), i, dtype=torch.long))
        labels.append(label)

    flat_images = torch.cat(images_list, dim=0)  # (Total_Imgs, 3, H, W)
    flat_indices = torch.cat(batch_indices, dim=0)  # (Total_Imgs,)

    # Labels can be int (category_id) or int64 (_id)
    labels_tensor = torch.tensor(labels, dtype=torch.int64)

    return flat_images, flat_indices, labels_tensor


# ==========================================
# FEATURE EXTRACTOR
# ==========================================


class FeatureExtractor:
    def __init__(self, debug_size=None):
        self.device = Config.DEVICE
        self.debug_size = debug_size
        self.mapper = HierarchyMapper(load_cached_data=True)

    def _get_models(self):
        """
        Loads frozen ResNet50 and EfficientNetB0 backbones.
        """
        # ResNet50
        resnet = models.resnet50(weights="DEFAULT")
        resnet.fc = nn.Identity()  # Output: 2048
        resnet.to(self.device)
        resnet.eval()

        # EfficientNet B0
        effnet = models.efficientnet_b0(weights="DEFAULT")
        effnet.classifier = nn.Identity()  # Output: 1280 (after avgpool)
        effnet.to(self.device)
        effnet.eval()

        return resnet, effnet

    def _process_split(self, name, metadata_path, bson_path, is_test=False):
        """
        Runs inference on a dataset split and returns aggregated features.
        """
        print(f"Extracting features for split: {name}...")

        # Load metadata
        df = pd.read_csv(metadata_path)
        if self.debug_size:
            print(f"Debug mode: limiting {name} to {self.debug_size} samples.")
            df = df.iloc[: self.debug_size]

        dataset = BSONDataset(df, bson_path, transform=TRANSFORM, is_test=is_test)

        # DataLoader
        # Batch size refers to number of products.
        # 128 products * ~2 imgs = 256 imgs per batch, fits easily in A100.
        loader = DataLoader(
            dataset,
            batch_size=128,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            collate_fn=collate_fn,
            pin_memory=True,
        )

        resnet, effnet = self._get_models()

        all_resnet_feats = []
        all_effnet_feats = []
        all_labels = []

        with torch.no_grad():
            for imgs, indices, labels in loader:
                imgs = imgs.to(self.device)
                indices = indices.to(self.device)

                # Forward Pass
                r_feats = resnet(imgs)  # (Total_Imgs, 2048)
                e_feats = effnet(imgs)  # (Total_Imgs, 1280)

                # Mean Pooling per Product
                batch_size = labels.shape[0]
                r_pooled = scatter_mean(r_feats, indices, dim=0, dim_size=batch_size)
                e_pooled = scatter_mean(e_feats, indices, dim=0, dim_size=batch_size)

                # Store (move to CPU to save GPU memory)
                all_resnet_feats.append(r_pooled.cpu().numpy())
                all_effnet_feats.append(e_pooled.cpu().numpy())
                all_labels.append(labels.numpy())

        # Concatenate all batches
        final_resnet = np.concatenate(all_resnet_feats, axis=0)
        final_effnet = np.concatenate(all_effnet_feats, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)

        return final_resnet, final_effnet, final_labels

    def extract_all(self, load_cached_data=True):
        """
        Main entry point. Checks cache, runs extraction, and saves NPY files.
        """
        # Define file groups for checking existence
        train_files = [
            Config.TRAIN_FEATS_RESNET,
            Config.TRAIN_FEATS_EFFNET,
            Config.TRAIN_LABELS,
        ]
        val_files = [
            Config.VAL_FEATS_RESNET,
            Config.VAL_FEATS_EFFNET,
            Config.VAL_LABELS,
        ]
        test_files = [
            Config.TEST_FEATS_RESNET,
            Config.TEST_FEATS_EFFNET,
            Config.TEST_IDS,
        ]

        all_files = train_files + val_files + test_files

        # Check Cache
        if load_cached_data and all(os.path.exists(f) for f in all_files):
            print("All features found in cache. Skipping extraction.")
            return

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # ---------------------------
        # PROCESS TRAIN
        # ---------------------------
        if not (load_cached_data and all(os.path.exists(f) for f in train_files)):
            r_feats, e_feats, cats = self._process_split(
                "train", Config.TRAIN_META, Config.TRAIN_BSON, is_test=False
            )

            # Convert category_id to hierarchical labels [L1, L2, L3]
            l3_indices = np.array([self.mapper.l3_id_to_idx.get(c, -1) for c in cats])
            l1_indices, l2_indices = self.mapper.get_parent_labels(l3_indices)
            labels_stacked = np.stack(
                [l1_indices, l2_indices, l3_indices], axis=1
            ).astype(np.int64)

            np.save(Config.TRAIN_FEATS_RESNET, r_feats)
            np.save(Config.TRAIN_FEATS_EFFNET, e_feats)
            np.save(Config.TRAIN_LABELS, labels_stacked)

            # Free memory
            del (
                r_feats,
                e_feats,
                cats,
                labels_stacked,
                l3_indices,
                l1_indices,
                l2_indices,
            )

        # ---------------------------
        # PROCESS VAL
        # ---------------------------
        if not (load_cached_data and all(os.path.exists(f) for f in val_files)):
            r_feats, e_feats, cats = self._process_split(
                "val", Config.VAL_META, Config.TRAIN_BSON, is_test=False
            )

            l3_indices = np.array([self.mapper.l3_id_to_idx.get(c, -1) for c in cats])
            l1_indices, l2_indices = self.mapper.get_parent_labels(l3_indices)
            labels_stacked = np.stack(
                [l1_indices, l2_indices, l3_indices], axis=1
            ).astype(np.int64)

            np.save(Config.VAL_FEATS_RESNET, r_feats)
            np.save(Config.VAL_FEATS_EFFNET, e_feats)
            np.save(Config.VAL_LABELS, labels_stacked)

            del r_feats, e_feats, cats, labels_stacked

        # ---------------------------
        # PROCESS TEST
        # ---------------------------
        if not (load_cached_data and all(os.path.exists(f) for f in test_files)):
            r_feats, e_feats, ids = self._process_split(
                "test", Config.TEST_META, Config.TEST_BSON, is_test=True
            )

            np.save(Config.TEST_FEATS_RESNET, r_feats)
            np.save(Config.TEST_FEATS_EFFNET, e_feats)
            np.save(Config.TEST_IDS, ids)

            del r_feats, e_feats, ids

        print("Feature extraction complete.")

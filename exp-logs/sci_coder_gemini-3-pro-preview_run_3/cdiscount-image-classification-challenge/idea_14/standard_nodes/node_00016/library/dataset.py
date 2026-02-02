import os
import struct
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from library.configuration import Config
from library.utilities import HierarchyManager


# ==========================================
# HELPER FUNCTIONS (BSON PARSING)
# ==========================================
def get_val_size(type_byte, data, ptr):
    """Returns the size of a BSON value based on its type byte."""
    if type_byte == 0x01:  # double
        return 8
    elif type_byte == 0x02:  # string
        s_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + s_len
    elif type_byte == 0x03:  # document
        d_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return d_len
    elif type_byte == 0x04:  # array
        a_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return a_len
    elif type_byte == 0x05:  # binary
        b_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
        return 4 + 1 + b_len
    elif type_byte == 0x07:  # objectid
        return 12
    elif type_byte == 0x08:  # boolean
        return 1
    elif type_byte == 0x09:  # utc datetime
        return 8
    elif type_byte == 0x0A:  # null
        return 0
    elif type_byte == 0x10:  # int32
        return 4
    elif type_byte == 0x12:  # int64
        return 8
    else:
        return 0


def extract_images_from_bson(data):
    """
    Parses a raw BSON document to find the 'imgs' array and extract 'picture' binaries.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        if name_end == -1:
            break
        name = data[ptr:name_end].decode("utf-8", errors="ignore")
        ptr = name_end + 1

        if name == "imgs" and type_byte == 0x04:
            # Found 'imgs' array
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len

            # Enter Array (skip length int)
            ap = ptr + 4
            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1

                # Array keys are "0", "1"... skip them
                ename_end = data.find(b"\x00", ap)
                if ename_end == -1:
                    break
                ap = ename_end + 1

                if etype == 0x03:  # Document (Image container)
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len

                    # Enter Document
                    dp = ap + 4
                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1

                        dname_end = data.find(b"\x00", dp)
                        if dname_end == -1:
                            break
                        dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        dp = dname_end + 1

                        if dname == "picture" and dtype == 0x05:
                            # Found picture binary
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype is at dp+4, data starts at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            # Skip other fields in image doc
                            v_len = get_val_size(dtype, data, dp)
                            dp += v_len

                    ap += doc_len
                else:
                    v_len = get_val_size(etype, data, ap)
                    ap += v_len

            ptr += arr_len
        else:
            # Skip this field
            v_len = get_val_size(type_byte, data, ptr)
            ptr += v_len

    return images


# ==========================================
# DATASET CLASSES
# ==========================================


class BSONRawImageDataset(Dataset):
    """
    Dataset for reading raw images from BSON files using metadata offsets.
    Used for the Feature Extraction phase.
    """

    def __init__(self, split="train", debug=False):
        self.split = split
        self.debug = debug
        self.hierarchy_manager = HierarchyManager()

        # Determine paths
        if split == "train":
            self.meta_path = Config.TRAIN_META
            self.bson_path = Config.TRAIN_BSON
        elif split == "val":
            self.meta_path = Config.VAL_META
            self.bson_path = Config.TRAIN_BSON
        elif split == "test":
            self.meta_path = Config.TEST_META
            self.bson_path = Config.TEST_BSON
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load Metadata
        self.meta_df = pd.read_csv(self.meta_path)

        if self.debug:
            self.meta_df = self.meta_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Transformations
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

        # File handle (initialized lazily in workers)
        self.file_handle = None

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if self.file_handle is None:
            self.file_handle = open(self.bson_path, "rb")

        row = self.meta_df.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        prod_id = row["_id"]

        # Read BSON
        self.file_handle.seek(offset)
        doc_data = self.file_handle.read(length)

        # Extract Images
        img_binaries = extract_images_from_bson(doc_data)

        images_tensors = []
        for img_bytes in img_binaries:
            # Decode using OpenCV
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform
            img_t = self.transform(img)
            images_tensors.append(img_t)

        # Handle case with no valid images (should be rare)
        if len(images_tensors) == 0:
            # Return a black image
            images_tensors.append(torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE))

        # Stack images: (K, C, H, W)
        images_stack = torch.stack(images_tensors)

        # Get Labels
        l3_idx = -1
        if self.split in ["train", "val"]:
            cat_id = int(row["category_id"])
            l3_idx = self.hierarchy_manager.cat_id_to_l3_idx.get(cat_id, -1)

        return images_stack, prod_id, l3_idx


def bson_collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.
    Flattens the images into a single batch for efficient CNN inference.

    Returns:
        batch_images: (Sum_K, C, H, W)
        batch_ids: List of product IDs
        batch_sizes: Tensor of number of images per product (for un-pooling)
        batch_l3_indices: Tensor of L3 labels
    """
    images_list = []
    ids_list = []
    sizes_list = []
    labels_list = []

    for images, prod_id, l3_idx in batch:
        images_list.append(images)
        ids_list.append(prod_id)
        sizes_list.append(images.shape[0])
        labels_list.append(l3_idx)

    batch_images = torch.cat(images_list, dim=0)
    batch_sizes = torch.tensor(sizes_list, dtype=torch.long)
    batch_l3_indices = torch.tensor(labels_list, dtype=torch.long)

    return batch_images, ids_list, batch_sizes, batch_l3_indices


class FeatureMemoryDataset(Dataset):
    """
    Dataset for loading pre-computed features and hierarchical labels from RAM.
    Used for the Cascade MLP Training phase.
    """

    def __init__(self, split="train", debug=False):
        self.split = split
        self.debug = debug
        self.hierarchy_manager = HierarchyManager()

        # Determine paths
        if split == "train":
            self.feat_path = Config.TRAIN_FEATURES_PATH
            self.label_path = Config.TRAIN_LABELS_PATH
            self.id_path = None
        elif split == "val":
            self.feat_path = Config.VAL_FEATURES_PATH
            self.label_path = Config.VAL_LABELS_PATH
            self.id_path = None
        elif split == "test":
            self.feat_path = Config.TEST_FEATURES_PATH
            self.label_path = None
            self.id_path = Config.TEST_IDS_PATH
        else:
            raise ValueError(f"Unknown split: {split}")

        # Load Data
        if not os.path.exists(self.feat_path):
            raise FileNotFoundError(
                f"Feature file not found: {self.feat_path}. Run feature extraction first."
            )

        # Load features into memory
        self.features = np.load(self.feat_path)

        if self.label_path:
            self.labels_l3 = np.load(self.label_path)
            # Pre-compute L1 and L2 labels for speed
            # Vectorized lookup
            v_l3_to_l2 = np.vectorize(
                lambda x: self.hierarchy_manager.l3_idx_to_l2_idx.get(x, -1)
            )
            v_l3_to_l1 = np.vectorize(
                lambda x: self.hierarchy_manager.l3_idx_to_l1_idx.get(x, -1)
            )

            self.labels_l2 = v_l3_to_l2(self.labels_l3).astype(np.int64)
            self.labels_l1 = v_l3_to_l1(self.labels_l3).astype(np.int64)
            self.ids = None
        else:
            self.labels_l3 = None
            self.labels_l2 = None
            self.labels_l1 = None
            if self.id_path:
                self.ids = np.load(self.id_path)
            else:
                self.ids = np.zeros(len(self.features), dtype=np.int64)

        if self.debug:
            limit = Config.DEBUG_SAMPLE_SIZE
            self.features = self.features[:limit]
            if self.labels_l3 is not None:
                self.labels_l3 = self.labels_l3[:limit]
                self.labels_l2 = self.labels_l2[:limit]
                self.labels_l1 = self.labels_l1[:limit]
            if self.ids is not None:
                self.ids = self.ids[:limit]

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Features
        feat = torch.from_numpy(self.features[idx]).float()

        if self.split in ["train", "val"]:
            l1 = torch.tensor(self.labels_l1[idx], dtype=torch.long)
            l2 = torch.tensor(self.labels_l2[idx], dtype=torch.long)
            l3 = torch.tensor(self.labels_l3[idx], dtype=torch.long)
            return feat, l1, l2, l3
        else:
            prod_id = self.ids[idx]
            return feat, prod_id


def get_extraction_dataloader(split="train", debug=False):
    """Factory for feature extraction dataloader."""
    dataset = BSONRawImageDataset(split=split, debug=debug)
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE_EXTRACT,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=bson_collate_fn,
        pin_memory=True,
    )


def get_training_dataloader(split="train", debug=False):
    """Factory for MLP training dataloader."""
    dataset = FeatureMemoryDataset(split=split, debug=debug)
    shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

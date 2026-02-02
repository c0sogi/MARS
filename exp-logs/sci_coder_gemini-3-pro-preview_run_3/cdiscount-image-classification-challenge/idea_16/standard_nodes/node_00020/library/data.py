import os
import struct
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from library.config import Config
from library.utils import HierarchyMapper


# ==========================================
# HELPER: BSON PARSING
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
    Returns a list of bytes objects.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        # name = data[ptr:name_end].decode("utf-8", errors="ignore") # Optimization: Don't decode unless needed
        # We only care if name is "imgs"
        is_imgs = (name_end - ptr == 4) and (data[ptr:name_end] == b"imgs")
        ptr = name_end + 1

        if is_imgs and type_byte == 0x04:
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
                        # dname = data[dp:dname_end].decode("utf-8", errors="ignore")
                        is_picture = (dname_end - dp == 7) and (
                            data[dp:dname_end] == b"picture"
                        )
                        dp = dname_end + 1

                        if is_picture and dtype == 0x05:
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
# DATASET: RAW IMAGE EXTRACTION
# ==========================================
class BSONImageDataset(Dataset):
    """
    Dataset for reading raw images from BSON files using metadata offsets.
    Used for the feature extraction phase.
    """

    def __init__(self, metadata_path, bson_path, transform=None, subset_size=None):
        self.bson_path = bson_path
        self.transform = transform

        # Load metadata
        self.meta = pd.read_csv(metadata_path)

        # Debugging subset
        if subset_size:
            self.meta = self.meta.iloc[:subset_size]

        # Pre-convert columns to numpy for faster access
        self.offsets = self.meta["bson_offset"].values
        self.lengths = self.meta["bson_length"].values
        self.ids = self.meta["_id"].values

        # Handle category_id if it exists (Train/Val), else None (Test)
        if "category_id" in self.meta.columns:
            self.category_ids = self.meta["category_id"].values
        else:
            self.category_ids = None

        # File handle (lazy initialization per worker)
        self.f = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        if self.f is None:
            self.f = open(self.bson_path, "rb")

        offset = self.offsets[idx]
        length = self.lengths[idx]

        # Seek and read
        self.f.seek(offset)
        doc_data = self.f.read(length)

        # Parse images
        img_binaries = extract_images_from_bson(doc_data)

        images = []
        for img_bytes in img_binaries:
            # Decode
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

            if img is None:
                continue

            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Transform
            if self.transform:
                img = self.transform(img)
            else:
                # Basic to tensor if no transform provided
                img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

            images.append(img)

        # Handle case with no valid images (rare but possible)
        if not images:
            # Create a black image
            img = torch.zeros(
                (3, Config.IMG_SIZE, Config.IMG_SIZE), dtype=torch.float32
            )
            images.append(img)

        # Stack images: (N, C, H, W)
        images_tensor = torch.stack(images)

        product_id = self.ids[idx]
        category_id = self.category_ids[idx] if self.category_ids is not None else -1

        return images_tensor, product_id, category_id


def product_collate_fn(batch):
    """
    Custom collate function to handle variable number of images per product.

    Args:
        batch: List of tuples (images_tensor, product_id, category_id)
               images_tensor shape: (N_imgs, C, H, W)

    Returns:
        all_images: (Total_Imgs, C, H, W) - Flattened batch of images
        sizes: (Batch_Size,) - Number of images per product
        product_ids: (Batch_Size,)
        category_ids: (Batch_Size,)
    """
    images_list = []
    sizes = []
    product_ids = []
    category_ids = []

    for imgs, pid, cid in batch:
        images_list.append(imgs)
        sizes.append(imgs.shape[0])
        product_ids.append(pid)
        category_ids.append(cid)

    all_images = torch.cat(images_list, dim=0)
    sizes = torch.tensor(sizes, dtype=torch.long)
    product_ids = torch.tensor(product_ids, dtype=torch.long)
    category_ids = torch.tensor(category_ids, dtype=torch.long)

    return all_images, sizes, product_ids, category_ids


# ==========================================
# DATASET: CACHED FEATURES
# ==========================================
class CachedTensorDataset(Dataset):
    """
    Dataset for training on pre-computed features stored in .npy files.
    Maps raw category_ids to hierarchical labels on-the-fly.
    """

    def __init__(
        self,
        features_path,
        labels_path=None,
        ids_path=None,
        hierarchy_mapper=None,
        subset_size=None,
    ):
        self.features_path = features_path
        self.labels_path = labels_path
        self.ids_path = ids_path
        self.mapper = hierarchy_mapper

        # Load data using mmap for memory efficiency
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        self.features = np.load(features_path, mmap_mode="r")

        self.labels = None
        if labels_path and os.path.exists(labels_path):
            self.labels = np.load(labels_path, mmap_mode="r")

        self.ids = None
        if ids_path and os.path.exists(ids_path):
            self.ids = np.load(ids_path, mmap_mode="r")

        # Determine length
        self.length = self.features.shape[0]
        if subset_size:
            self.length = min(self.length, subset_size)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Load feature vector
        # Copy to ensure it's a writable array (needed for some torch ops) and not a memmap view
        x = torch.from_numpy(np.array(self.features[idx])).float()

        if self.labels is not None:
            # Training/Validation mode
            raw_cat_id = self.labels[idx]

            # Get hierarchical labels
            # We wrap in list to use the mapper's batch function, then extract
            mapping = self.mapper.get_labels([raw_cat_id])

            l1 = torch.tensor(mapping["l1"][0], dtype=torch.long)
            l2 = torch.tensor(mapping["l2"][0], dtype=torch.long)
            l3 = torch.tensor(mapping["l3"][0], dtype=torch.long)

            return x, l1, l2, l3
        else:
            # Inference mode
            prod_id = self.ids[idx] if self.ids is not None else -1
            return x, prod_id


# ==========================================
# FACTORY FUNCTIONS
# ==========================================
def get_extraction_loader(metadata_path, bson_path, subset_size=None):
    """
    Creates a DataLoader for the feature extraction phase.
    """
    # Define Transforms
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=Config.IMG_MEAN, std=Config.IMG_STD),
        ]
    )

    dataset = BSONImageDataset(
        metadata_path=metadata_path,
        bson_path=bson_path,
        transform=transform,
        subset_size=subset_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE_EXTRACTION,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=product_collate_fn,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader


def get_training_loader(
    features_path, labels_path, hierarchy_mapper, shuffle=True, subset_size=None
):
    """
    Creates a DataLoader for the training phase using cached features.
    """
    dataset = CachedTensorDataset(
        features_path=features_path,
        labels_path=labels_path,
        hierarchy_mapper=hierarchy_mapper,
        subset_size=subset_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,
        shuffle=shuffle,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader


def get_test_loader(features_path, ids_path, subset_size=None):
    """
    Creates a DataLoader for the inference phase.
    """
    dataset = CachedTensorDataset(
        features_path=features_path, ids_path=ids_path, subset_size=subset_size
    )

    loader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE_TRAIN,  # Can use same large batch size as training
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    return loader

import os
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import timm
import cv2
from library.config import Config
from library.utils import seed_everything


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
    Returns a list of byte strings.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1

        # Read Field Name
        name_end = data.find(b"\x00", ptr)
        # name = data[ptr:name_end].decode("utf-8", errors="ignore")
        # We know we are looking for 'imgs', we can check bytes directly or decode
        # Optimization: just check if it matches b'imgs'
        is_imgs = data[ptr:name_end] == b"imgs"
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
                        dname_bytes = data[dp:dname_end]
                        dp = dname_end + 1

                        if dname_bytes == b"picture" and dtype == 0x05:
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
# DATASET CLASS
# ==========================================
class BSONProductDataset(Dataset):
    def __init__(self, metadata_path, bson_dir, transform=None, is_test=False):
        """
        Args:
            metadata_path: Path to the CSV containing _id, bson_offset, bson_length, file_path
            bson_dir: Directory containing the .bson files
            transform: PyTorch transforms
            is_test: Boolean, if True, returns dummy label
        """
        self.meta = pd.read_csv(metadata_path)
        self.bson_dir = bson_dir
        self.transform = transform
        self.is_test = is_test

        # Pre-compute full paths to avoid joining strings in loop
        self.meta["full_path"] = self.meta["file_path"].apply(
            lambda x: os.path.join(bson_dir, x)
        )

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        file_path = row["full_path"]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read BSON chunk
        with open(file_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        # Extract Images
        img_binaries = extract_images_from_bson(data)

        images = []
        for img_bytes in img_binaries:
            # Decode
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if self.transform:
                img = self.transform(img)

            images.append(img)

        # Handle case with no valid images
        if not images:
            # Return a black image
            dummy = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)
            images.append(dummy)

        # Stack images: (N_imgs, 3, H, W)
        images_tensor = torch.stack(images)

        # Label/ID
        if self.is_test:
            label = -1
        else:
            label = row["category_id"]

        product_id = row["_id"]

        return images_tensor, label, product_id


def collate_product_batch(batch):
    """
    Custom collate to handle variable number of images per product.
    Flattens the batch into a single large tensor of images, keeping track of counts.
    """
    batch_imgs = []
    batch_counts = []
    batch_labels = []
    batch_ids = []

    for imgs, label, pid in batch:
        batch_imgs.append(imgs)
        batch_counts.append(imgs.shape[0])
        batch_labels.append(label)
        batch_ids.append(pid)

    # Concatenate all images into one batch: (Sum_N, 3, H, W)
    combined_imgs = torch.cat(batch_imgs, dim=0)

    return (
        combined_imgs,
        torch.tensor(batch_counts),
        np.array(batch_labels),
        np.array(batch_ids),
    )


# ==========================================
# FEATURE EXTRACTOR
# ==========================================
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Transforms
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

        # Initialize Models
        print("Initializing Dual Backbones...")

        # 1. ResNet50 (2048 dim)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Remove classification layer
        self.resnet.to(self.device)
        self.resnet.eval()

        # 2. EfficientNet-B0 (1280 dim)
        self.effnet = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        self.effnet.to(self.device)
        self.effnet.eval()

    def _process_loader(self, loader, total_samples, desc="Extracting"):
        """
        Runs inference on the loader and returns pooled features, labels, and ids.
        """
        # Pre-allocate memory
        # Feature dim: ResNet(2048) + EffNet(1280) = 3328
        feature_dim = Config.INPUT_DIM

        all_features = np.zeros((total_samples, feature_dim), dtype=np.float32)
        all_labels = np.zeros(total_samples, dtype=np.int64)
        all_ids = np.zeros(total_samples, dtype=np.int64)

        ptr = 0

        with torch.no_grad():
            for imgs, counts, labels, ids in loader:
                imgs = imgs.to(self.device)

                # Forward Pass - ResNet
                f_res = self.resnet(imgs)  # (Total_K, 2048)

                # Forward Pass - EfficientNet
                f_eff = self.effnet(imgs)  # (Total_K, 1280)

                # Concatenate
                features = torch.cat([f_res, f_eff], dim=1)  # (Total_K, 3328)

                # Mean Pool per Product
                # Split features back into product groups
                split_features = torch.split(features, counts.tolist())

                pooled_list = []
                for f in split_features:
                    pooled_list.append(f.mean(dim=0))

                batch_pooled = torch.stack(pooled_list).cpu().numpy()

                # Store
                batch_size = batch_pooled.shape[0]
                all_features[ptr : ptr + batch_size] = batch_pooled
                all_labels[ptr : ptr + batch_size] = labels
                all_ids[ptr : ptr + batch_size] = ids

                ptr += batch_size

                if ptr % 10000 == 0:
                    print(f"{desc}: Processed {ptr}/{total_samples} samples")

        return all_features, all_labels, all_ids

    def extract_features(self, load_cached_data=True):
        """
        Main method to extract features for Train, Val, and Test sets.
        Handles caching logic.
        """
        seed_everything()
        os.makedirs(Config.WORKING_DIR, exist_ok=True)

        # Define tasks: (Name, MetadataPath, FeaturePath, LabelPath, IdPath, IsTest)
        tasks = [
            (
                "Train",
                Config.TRAIN_META,
                Config.TRAIN_FEATURES_PATH,
                Config.TRAIN_LABELS_PATH,
                None,
                False,
            ),
            (
                "Val",
                Config.VAL_META,
                Config.VAL_FEATURES_PATH,
                Config.VAL_LABELS_PATH,
                None,
                False,
            ),
            (
                "Test",
                Config.TEST_META,
                Config.TEST_FEATURES_PATH,
                None,
                Config.TEST_IDS_PATH,
                True,
            ),
        ]

        for name, meta_path, feat_path, label_path, id_path, is_test in tasks:
            print(f"\nChecking cache for {name} features...")

            # Check if cache exists
            cache_exists = os.path.exists(feat_path)
            if label_path:
                cache_exists = cache_exists and os.path.exists(label_path)
            if id_path:
                cache_exists = cache_exists and os.path.exists(id_path)

            if load_cached_data and cache_exists:
                print(f"Cache found for {name}. Skipping extraction.")
                continue

            print(
                f"Cache missing or reload requested. Starting extraction for {name}..."
            )

            # Load Metadata
            if not os.path.exists(meta_path):
                print(f"Error: Metadata file {meta_path} not found.")
                continue

            # Debugging: subset if Config.DEBUG is True
            if Config.DEBUG:
                print(f"DEBUG MODE: Limiting {name} to {Config.DEBUG_SIZE} samples.")
                # We read csv with nrows
                # But we need to shuffle or just take top? Top is fine for feature extraction test.
                # Actually, Dataset reads full csv. Let's handle it by slicing the dataframe inside Dataset if needed,
                # but here we can just pass a modified csv or handle in Dataset.
                # Simpler: Modify Dataset to accept a limit? No, let's just use full for now,
                # or read dataframe here and pass to Dataset.
                pass

            # Create Dataset and Loader
            dataset = BSONProductDataset(
                metadata_path=meta_path,
                bson_dir=Config.INPUT_DIR,
                transform=self.transform,
                is_test=is_test,
            )

            if Config.DEBUG:
                dataset.meta = dataset.meta.head(Config.DEBUG_SIZE)

            loader = DataLoader(
                dataset,
                batch_size=128,  # Products per batch (images will be ~1.5x this)
                shuffle=False,
                num_workers=Config.NUM_WORKERS,
                collate_fn=collate_product_batch,
                pin_memory=True,
            )

            # Extract
            feats, labels, ids = self._process_loader(loader, len(dataset), desc=name)

            # Save
            print(f"Saving {name} artifacts...")
            np.save(feat_path, feats)
            print(f"Saved features to {feat_path} shape={feats.shape}")

            if label_path:
                np.save(label_path, labels)
                print(f"Saved labels to {label_path}")

            if id_path:
                np.save(id_path, ids)
                print(f"Saved IDs to {id_path}")

        print("\nFeature extraction pipeline completed.")

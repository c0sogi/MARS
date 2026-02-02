import os
import struct
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import cv2
from torch_scatter import scatter
from library.config import Config
from library.utils import seed_everything


# ==========================================
# HELPER FUNCTIONS (BSON PARSING)
# ==========================================
def get_val_size(type_byte, data, ptr):
    """Returns the size of a BSON value based on its type byte."""
    if type_byte == 0x01:
        return 8
    elif type_byte == 0x02:
        return 4 + struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x03:
        return struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x04:
        return struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x05:
        return 4 + 1 + struct.unpack("<i", data[ptr : ptr + 4])[0]
    elif type_byte == 0x07:
        return 12
    elif type_byte == 0x08:
        return 1
    elif type_byte == 0x09:
        return 8
    elif type_byte == 0x0A:
        return 0
    elif type_byte == 0x10:
        return 4
    elif type_byte == 0x12:
        return 8
    return 0


def extract_images_from_bytes(data):
    """
    Parses a BSON byte string to find and extract image binaries.
    """
    images = []
    ptr = 4  # Skip total size header
    length = len(data)

    while ptr < length - 1:
        type_byte = data[ptr]
        ptr += 1
        name_end = data.find(b"\x00", ptr)
        # name = data[ptr:name_end].decode("utf-8", errors="ignore") # Optimization: Don't decode unless needed
        # We only care about "imgs"
        is_imgs = data[ptr:name_end] == b"imgs"
        ptr = name_end + 1

        if is_imgs and type_byte == 0x04:
            arr_len = struct.unpack("<i", data[ptr : ptr + 4])[0]
            arr_end = ptr + arr_len
            ap = ptr + 4
            while ap < arr_end - 1:
                etype = data[ap]
                ap += 1
                ename_end = data.find(b"\x00", ap)
                ap = ename_end + 1
                if etype == 0x03:  # Document
                    doc_len = struct.unpack("<i", data[ap : ap + 4])[0]
                    doc_end = ap + doc_len
                    dp = ap + 4
                    while dp < doc_end - 1:
                        dtype = data[dp]
                        dp += 1
                        dname_end = data.find(b"\x00", dp)
                        is_picture = data[dp:dname_end] == b"picture"
                        dp = dname_end + 1
                        if is_picture and dtype == 0x05:
                            bin_len = struct.unpack("<i", data[dp : dp + 4])[0]
                            # subtype at dp+4, data at dp+5
                            img_bytes = data[dp + 5 : dp + 5 + bin_len]
                            images.append(img_bytes)
                            dp += 4 + 1 + bin_len
                        else:
                            dp += get_val_size(dtype, data, dp)
                    ap += doc_len
                else:
                    ap += get_val_size(etype, data, ap)
            ptr += arr_len
        else:
            ptr += get_val_size(type_byte, data, ptr)
    return images


# ==========================================
# DATASET & DATALOADER
# ==========================================
class BSONDataset(Dataset):
    def __init__(self, metadata_df, bson_path, transform=None, is_test=False):
        self.meta = metadata_df
        self.bson_path = bson_path
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read raw BSON data
        with open(self.bson_path, "rb") as f:
            f.seek(offset)
            data = f.read(length)

        # Extract image bytes
        img_binaries = extract_images_from_bytes(data)

        processed_imgs = []
        if len(img_binaries) == 0:
            # Fallback for missing images: Black image
            processed_imgs.append(torch.zeros(3, 224, 224))
        else:
            for img_bytes in img_binaries:
                # Decode
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                # Convert BGR to RGB
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply Transforms
                if self.transform:
                    img = self.transform(img)
                processed_imgs.append(img)

        if len(processed_imgs) == 0:
            processed_imgs.append(torch.zeros(3, 224, 224))

        # Stack images: (K, C, H, W)
        imgs_tensor = torch.stack(processed_imgs)

        # Get Label or ID
        if self.is_test:
            target = row["_id"]
        else:
            target = row["category_id"]

        return imgs_tensor, target


def collate_fn(batch):
    """
    Custom collator to handle variable number of images per product.
    Flattens images into a single batch and creates an index mapping.
    """
    batch_imgs = []
    batch_targets = []
    batch_indices = []

    for i, (imgs, target) in enumerate(batch):
        batch_imgs.append(imgs)
        batch_targets.append(target)
        # Map these K images to product index i in the batch
        batch_indices.extend([i] * imgs.size(0))

    # Concatenate all images: (Total_Images, C, H, W)
    cat_imgs = torch.cat(batch_imgs, dim=0)

    return (
        cat_imgs,
        torch.tensor(batch_targets),
        torch.tensor(batch_indices, dtype=torch.long),
    )


# ==========================================
# EXTRACTION LOGIC
# ==========================================
def extract_features_to_disk(
    metadata_path,
    bson_path,
    out_feat_path,
    out_label_path,
    is_test=False,
    debug_size=None,
):
    """
    Runs the ResNet50 feature extraction and saves to disk.
    """
    print(f"Starting feature extraction for {os.path.basename(metadata_path)}...")

    # 1. Load Metadata
    df = pd.read_csv(metadata_path)
    if debug_size:
        df = df.head(debug_size)
        print(f"Debug mode: Processing only {len(df)} records.")

    # 2. Setup Data
    # Standard ImageNet normalization
    transform = T.Compose(
        [
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    dataset = BSONDataset(df, bson_path, transform=transform, is_test=is_test)

    # Use a safe batch size for Image Inference (A100 can handle ~256-512 ResNet50 images)
    # Config.BATCH_SIZE is 2048 (for MLPs), which is too big for images.
    IMG_BATCH_SIZE = 256

    loader = DataLoader(
        dataset,
        batch_size=IMG_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 3. Setup Model
    device = torch.device(Config.DEVICE)
    model = models.resnet50(weights="DEFAULT")
    # Remove classification head (fc)
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    # 4. Pre-allocate Output Arrays
    num_samples = len(df)
    features_cache = np.zeros((num_samples, 2048), dtype=np.float32)
    labels_cache = np.zeros((num_samples,), dtype=np.int64)

    # 5. Inference Loop
    ptr = 0
    print(f"Processing {num_samples} products...")

    with torch.no_grad():
        for batch_idx, (imgs, targets, indices) in enumerate(loader):
            imgs = imgs.to(device)
            indices = indices.to(device)

            # Forward Pass (Mixed Precision for speed)
            with torch.cuda.amp.autocast():
                feats = model(imgs)  # (Total_Images, 2048)

            # Mean Pooling per Product
            # indices maps every image to a product index in the range [0, batch_size-1]
            # We want output (Batch_Size, 2048)
            # scatter(src, index, dim, reduce)
            # We must ensure the output size covers the full batch, even if some products had no images (handled in dataset)
            batch_size = targets.size(0)
            pooled_feats = scatter(
                feats, indices, dim=0, dim_size=batch_size, reduce="mean"
            )

            # Store in RAM
            batch_len = pooled_feats.size(0)
            features_cache[ptr : ptr + batch_len] = (
                pooled_feats.cpu().numpy().astype(np.float32)
            )
            labels_cache[ptr : ptr + batch_len] = targets.numpy()

            ptr += batch_len

            if (batch_idx + 1) % 100 == 0:
                print(f"Processed {ptr}/{num_samples} records.")

    # 6. Save to Disk
    print(f"Saving features to {out_feat_path}...")
    np.save(out_feat_path, features_cache)
    np.save(out_label_path, labels_cache)
    print("Save complete.")


def run_feature_extraction(load_cached_data=True, debug=False):
    """
    Main entry point for feature extraction.
    Checks cache, runs extraction if needed for Train, Val, and Test sets.
    """
    seed_everything()

    # Ensure cache directory exists
    os.makedirs(Config.CACHE_DIR, exist_ok=True)

    debug_size = 5000 if debug else None

    # -----------------------------------------------
    # 1. Train Set
    # -----------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.TRAIN_FEATURES_PATH)
        and os.path.exists(Config.TRAIN_LABELS_PATH)
    ):
        print("Loading cached Train features...")
    else:
        extract_features_to_disk(
            metadata_path=Config.TRAIN_META,
            bson_path=Config.TRAIN_BSON,
            out_feat_path=Config.TRAIN_FEATURES_PATH,
            out_label_path=Config.TRAIN_LABELS_PATH,
            is_test=False,
            debug_size=debug_size,
        )

    # -----------------------------------------------
    # 2. Validation Set
    # -----------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.VAL_FEATURES_PATH)
        and os.path.exists(Config.VAL_LABELS_PATH)
    ):
        print("Loading cached Val features...")
    else:
        extract_features_to_disk(
            metadata_path=Config.VAL_META,
            bson_path=Config.TRAIN_BSON,
            out_feat_path=Config.VAL_FEATURES_PATH,
            out_label_path=Config.VAL_LABELS_PATH,
            is_test=False,
            debug_size=debug_size,
        )

    # -----------------------------------------------
    # 3. Test Set
    # -----------------------------------------------
    if (
        load_cached_data
        and os.path.exists(Config.TEST_FEATURES_PATH)
        and os.path.exists(Config.TEST_IDS_PATH)
    ):
        print("Loading cached Test features...")
    else:
        extract_features_to_disk(
            metadata_path=Config.TEST_META,
            bson_path=Config.TEST_BSON,
            out_feat_path=Config.TEST_FEATURES_PATH,
            out_label_path=Config.TEST_IDS_PATH,
            is_test=True,
            debug_size=debug_size,
        )

    print("Feature extraction pipeline finished.")

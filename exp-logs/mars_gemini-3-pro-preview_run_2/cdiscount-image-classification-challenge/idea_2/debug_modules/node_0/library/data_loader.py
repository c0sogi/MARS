import os
import io
import struct
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import library.config as config

# ==== BSON Constants ====
TYPE_DOC = 3
TYPE_ARRAY = 4
TYPE_BINARY = 5


def read_cstring(buffer, offset):
    """Reads a null-terminated string from the buffer."""
    end = offset
    while end < len(buffer) and buffer[end] != 0:
        end += 1
    return buffer[offset:end].decode("utf-8", errors="ignore"), end + 1


def skip_value(buffer, offset, dtype):
    """Calculates the new offset after skipping a value of a given BSON type."""
    if dtype == 1:  # Double
        return offset + 8
    elif dtype == 2:  # String
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + l
    elif dtype == 3:  # Doc
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == 4:  # Array
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + l
    elif dtype == 5:  # Binary
        l = struct.unpack_from("<i", buffer, offset)[0]
        return offset + 4 + 1 + l
    elif dtype == 8:  # Bool
        return offset + 1
    elif dtype == 16:  # Int32
        return offset + 4
    elif dtype == 18:  # Int64
        return offset + 8
    elif dtype == 7:  # ObjectId
        return offset + 12
    elif dtype == 9:  # DateTime
        return offset + 8
    elif dtype == 10:  # Null
        return offset
    else:
        return offset


def extract_images_from_bytes(data):
    """
    Parses a raw BSON record bytes object and extracts image binary data.
    Returns a list of bytes objects (JPEG data).
    """
    images = []
    offset = 4  # Skip total size header
    length = len(data)

    while offset < length - 1:
        dtype = data[offset]
        offset += 1
        key, offset = read_cstring(data, offset)

        if key == "imgs" and dtype == TYPE_ARRAY:
            arr_size = struct.unpack_from("<i", data, offset)[0]
            arr_end = offset + arr_size
            offset += 4

            while offset < arr_end - 1:
                e_type = data[offset]
                offset += 1
                e_key, offset = read_cstring(data, offset)

                if e_type == TYPE_DOC:
                    doc_size = struct.unpack_from("<i", data, offset)[0]
                    doc_end = offset + doc_size

                    # Inside the image doc, look for 'picture'
                    sub_offset = offset + 4
                    while sub_offset < doc_end - 1:
                        s_type = data[sub_offset]
                        sub_offset += 1
                        s_key, sub_offset = read_cstring(data, sub_offset)

                        if s_key == "picture" and s_type == TYPE_BINARY:
                            b_len = struct.unpack_from("<i", data, sub_offset)[0]
                            sub_offset += 4
                            # subtype = data[sub_offset] # skip subtype read
                            sub_offset += 1
                            img_data = data[sub_offset : sub_offset + b_len]
                            images.append(img_data)
                            sub_offset += b_len
                        else:
                            sub_offset = skip_value(data, sub_offset, s_type)

                    offset = doc_end
                else:
                    offset = skip_value(data, offset, e_type)
        else:
            offset = skip_value(data, offset, dtype)

    return images


class BSONDataset(Dataset):
    def __init__(self, metadata_df, root_dir, transform=None, mode="train"):
        """
        Args:
            metadata_df (pd.DataFrame): DataFrame containing ['product_id', 'category_id', 'bson_offset', 'bson_length', 'file_path']
            root_dir (str): Directory containing the .bson files.
            transform (callable, optional): Transform to be applied on a sample.
            mode (str): 'train', 'val', or 'test'.
        """
        self.metadata = metadata_df
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.file_handles = {}  # Cache file handles per worker

        # Map category_id to class index (0 to NUM_CLASSES-1)
        # We need a consistent mapping.
        # Ideally, this should be provided or computed.
        # For this competition, category_ids are large integers.
        # We will assume the model handles mapping or we map here.
        # To ensure consistency across runs, we sort unique category_ids from the metadata if possible,
        # but usually, we need a fixed map.
        # For now, we will return the raw category_id (or mapped if we had a mapping file).
        # NOTE: The config specifies NUM_CLASSES = 5270.
        # We need to map the large category_id to 0-5269.
        # We will load the category_names.csv to build this map to ensure it covers all classes.

        cat_df = pd.read_csv(config.CATEGORY_NAMES)
        unique_cats = sorted(cat_df["category_id"].unique())
        self.cat_to_idx = {cat: i for i, cat in enumerate(unique_cats)}
        self.idx_to_cat = {i: cat for i, cat in enumerate(unique_cats)}

    def _get_handle(self, filename):
        if filename not in self.file_handles:
            path = os.path.join(self.root_dir, filename)
            self.file_handles[filename] = open(path, "rb")
        return self.file_handles[filename]

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        offset = row["bson_offset"]
        length = row["bson_length"]
        filename = row["file_path"]

        # Retrieve file handle
        f = self._get_handle(filename)

        # Read data
        f.seek(offset)
        data = f.read(length)

        # Extract images
        try:
            img_bytes_list = extract_images_from_bytes(data)
        except Exception:
            # Fallback for corrupt data (should not happen with verified metadata)
            img_bytes_list = []

        if len(img_bytes_list) == 0:
            # Create a black image if extraction fails
            images = [Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))]
        else:
            # Decode images
            images = []
            for b in img_bytes_list:
                try:
                    img = Image.open(io.BytesIO(b)).convert("RGB")
                    images.append(img)
                except:
                    pass
            if not images:
                images = [Image.new("RGB", (config.IMG_SIZE, config.IMG_SIZE))]

        # Mode-specific processing
        if self.mode == "train":
            # Select one image randomly
            img = images[np.random.randint(len(images))]
            if self.transform:
                img = self.transform(img)

            target = self.cat_to_idx.get(row["category_id"], 0)
            return img, target

        elif self.mode == "val":
            # Return all images stacked
            tensors = []
            for img in images:
                if self.transform:
                    tensors.append(self.transform(img))
                else:
                    tensors.append(transforms.ToTensor()(img))

            # Stack: (N, C, H, W)
            images_stack = torch.stack(tensors)
            target = self.cat_to_idx.get(row["category_id"], 0)
            return images_stack, target

        elif self.mode == "test":
            # Return all images stacked and product_id
            tensors = []
            for img in images:
                if self.transform:
                    tensors.append(self.transform(img))
                else:
                    tensors.append(transforms.ToTensor()(img))

            images_stack = torch.stack(tensors)
            product_id = row["product_id"]
            return images_stack, product_id


def get_transforms(mode="train"):
    if mode == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(config.IMG_SIZE, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(config.IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.MEAN, std=config.STD),
            ]
        )


def get_loaders(load_cached_data=True):
    """
    Creates DataLoaders for train, val, and test sets.

    Args:
        load_cached_data (bool): Unused, but kept for signature compatibility.
                                 Metadata loading is always cached via CSVs.
    """
    # Load Metadata
    print("Loading metadata...")
    df_train = pd.read_csv(config.TRAIN_META_CSV)
    df_val = pd.read_csv(config.VAL_META_CSV)
    df_test = pd.read_csv(config.TEST_META_CSV)

    # Debug Subsampling
    if config.DEBUG:
        print(f"DEBUG mode: Sampling {config.DEBUG_SAMPLE_SIZE} records.")
        df_train = df_train.sample(
            n=min(len(df_train), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )
        df_val = df_val.sample(
            n=min(len(df_val), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )
        df_test = df_test.sample(
            n=min(len(df_test), config.DEBUG_SAMPLE_SIZE), random_state=config.SEED
        )

    # Datasets
    train_dataset = BSONDataset(
        df_train, config.INPUT_DIR, transform=get_transforms("train"), mode="train"
    )

    val_dataset = BSONDataset(
        df_val, config.INPUT_DIR, transform=get_transforms("val"), mode="val"
    )

    test_dataset = BSONDataset(
        df_test, config.INPUT_DIR, transform=get_transforms("test"), mode="test"
    )

    # DataLoaders
    # Train: Standard batching
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
    )

    # Val/Test: Batch size 1 because each sample returns a tensor of shape (N, C, H, W)
    # where N varies per product. The model will treat N as the batch dimension during inference.
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    print(
        f"DataLoaders created. Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches."
    )
    return train_loader, val_loader, test_loader

import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from torchvision import transforms
from library import config, utils


class BSONInferenceDataset(Dataset):
    """
    Dataset for reading raw images from BSON files for feature extraction.
    It reads binary data using offsets and returns all images associated with a product ID
    as a stacked tensor.
    """

    def __init__(self, metadata, bson_path, transform=None):
        """
        Args:
            metadata (pd.DataFrame): Metadata containing _id, bson_offset, bson_length.
            bson_path (str): Path to the source BSON file.
            transform (callable, optional): PyTorch transforms. If None, uses default ImageNet stats.
        """
        self.metadata = metadata
        self.bson_path = bson_path
        self.reader = utils.BSONReader(bson_path)

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.ToPILImage(),
                    transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD
                    ),
                ]
            )
        else:
            self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Retrieve metadata for the record
        row = self.metadata.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]
        product_id = row["_id"]

        # Read binary image data
        try:
            img_bytes_list = self.reader.read_record(offset, length)
        except Exception as e:
            # Fallback for read errors
            print(f"Error reading record at index {idx}, offset {offset}: {e}")
            img_bytes_list = []

        images = []
        for img_bytes in img_bytes_list:
            try:
                # Decode image from bytes
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is None:
                    continue

                # Convert BGR (OpenCV) to RGB (PyTorch/PIL)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Apply transforms
                if self.transform:
                    img = self.transform(img)

                images.append(img)
            except Exception:
                continue

        # Handle case with no valid images (fallback to black image to prevent crash)
        if len(images) == 0:
            # Create a black image with correct shape (C, H, W)
            images = [
                torch.zeros((3, config.IMG_SIZE, config.IMG_SIZE), dtype=torch.float32)
            ]

        # Stack images into a single tensor: (Num_Images, C, H, W)
        images_tensor = torch.stack(images)

        return images_tensor, product_id


def inference_collate(batch):
    """
    Custom collate function to handle variable number of images per product.
    Used with DataLoader during feature extraction.

    Args:
        batch: List of tuples (images_tensor, product_id)
    Returns:
        images: List of image tensors (one tensor per product).
        ids: List of product IDs.
    """
    images = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    return images, ids


class EmbeddingDataset(Dataset):
    """
    Dataset for training the hierarchical MLP using pre-computed embeddings.
    Loads features and labels from .npy files into RAM.
    """

    def __init__(
        self,
        features_path,
        l1_path=None,
        l2_path=None,
        l3_path=None,
        ids_path=None,
        mode="train",
    ):
        """
        Args:
            features_path (str): Path to .npy file with product embeddings.
            l1_path (str): Path to L1 labels .npy (required for train/val).
            l2_path (str): Path to L2 labels .npy (required for train/val).
            l3_path (str): Path to L3 labels .npy (required for train/val).
            ids_path (str): Path to IDs .npy (required for test).
            mode (str): 'train', 'val', or 'test'.
        """
        self.mode = mode

        # Load features
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        # Load full array into memory
        self.features = np.load(features_path)

        if self.mode in ["train", "val"]:
            if not (l1_path and l2_path and l3_path):
                raise ValueError(
                    "L1, L2, and L3 paths must be provided for train/val mode."
                )

            self.l1_labels = np.load(l1_path)
            self.l2_labels = np.load(l2_path)
            self.l3_labels = np.load(l3_path)

            # Consistency check
            if not (
                len(self.features)
                == len(self.l1_labels)
                == len(self.l2_labels)
                == len(self.l3_labels)
            ):
                raise ValueError(f"Length mismatch in dataset files for {mode} split.")

        elif self.mode == "test":
            if ids_path:
                self.ids = np.load(ids_path)
                if len(self.features) != len(self.ids):
                    raise ValueError("Length mismatch between features and IDs.")
            else:
                self.ids = None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        # Return feature as float tensor
        feature = torch.from_numpy(self.features[idx]).float()

        if self.mode in ["train", "val"]:
            # Return labels as long tensors for CrossEntropyLoss
            l1 = torch.tensor(self.l1_labels[idx], dtype=torch.long)
            l2 = torch.tensor(self.l2_labels[idx], dtype=torch.long)
            l3 = torch.tensor(self.l3_labels[idx], dtype=torch.long)
            return feature, l1, l2, l3
        else:
            # Test mode
            if self.ids is not None:
                return feature, self.ids[idx]
            return feature

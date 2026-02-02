import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, Sampler
from sklearn.preprocessing import LabelEncoder

from library.config import Config


def get_transforms(data_type="train"):
    """
    Returns the Albumentations transformations for the specific data type.

    Args:
        data_type (str): 'train', 'val', or 'test'.
    """
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    p=0.2, brightness_limit=0.1, contrast_limit=0.1
                ),
                A.Normalize(
                    mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(),
            ]
        )
    else:
        # Validation and Test
        return A.Compose(
            [
                A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
                A.Normalize(
                    mean=Config.MEAN, std=Config.STD, max_pixel_value=255.0, p=1.0
                ),
                ToTensorV2(),
            ]
        )


def process_data(load_cached_data=True):
    """
    Loads metadata, handles label encoding, and manages caching of the encoder.

    Args:
        load_cached_data (bool): If True, attempts to load the label encoder from disk.
                                 If False or file missing, fits a new encoder and saves it.

    Returns:
        train_df (pd.DataFrame): Processed training metadata.
        val_df (pd.DataFrame): Processed validation metadata.
        test_df (pd.DataFrame): Processed test metadata.
        num_classes (int): Total number of unique hotel classes.
    """
    # Ensure working directory exists
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    encoder_path = os.path.join(Config.WORKING_DIR, "label_encoder.npy")

    # Load Metadata
    train_df = pd.read_csv(Config.TRAIN_CSV)
    val_df = pd.read_csv(Config.VAL_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)

    # Debugging: Subsample if configured
    if Config.DEBUG:
        train_df = train_df.sample(
            n=min(len(train_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            n=min(len(val_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)
        # Test set is usually small enough, but we can sample it too if needed
        test_df = test_df.sample(
            n=min(len(test_df), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Label Encoding Logic
    le = LabelEncoder()

    # Check cache
    if load_cached_data and os.path.exists(encoder_path):
        try:
            le.classes_ = np.load(encoder_path, allow_pickle=True)
        except Exception:
            # Fallback if load fails
            le.fit(train_df["hotel_id"])
            np.save(encoder_path, le.classes_)
    else:
        # Fit and save
        le.fit(train_df["hotel_id"])
        np.save(encoder_path, le.classes_)

    # Transform labels
    # Note: Validation set might contain classes not in training set if split wasn't perfect
    # (though our metadata script tries to handle this).
    # We handle unseen labels by filtering or assigning a dummy, but strictly speaking
    # for this task, we assume closed-set or we focus on known classes.
    # Here we filter val to ensure valid labels.

    train_df["label"] = le.transform(train_df["hotel_id"])

    # Filter validation set to only include classes seen in training
    valid_classes = set(le.classes_)
    val_df = val_df[val_df["hotel_id"].isin(valid_classes)].copy()
    val_df["label"] = le.transform(val_df["hotel_id"])

    num_classes = len(le.classes_)

    return train_df, val_df, test_df, num_classes


class HotelDataset(Dataset):
    def __init__(self, df, transforms=None, root_dir=Config.INPUT_DIR, is_test=False):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            transforms (albumentations.Compose): Transformations to apply.
            root_dir (str): Root directory for images.
            is_test (bool): If True, returns dummy label.
        """
        self.df = df
        self.transforms = transforms
        self.root_dir = root_dir
        self.is_test = is_test

        # Pre-compute file paths to avoid overhead in __getitem__
        self.file_paths = self.df["file_path"].values

        if not self.is_test:
            self.labels = self.df["label"].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        full_path = os.path.join(self.root_dir, file_path)

        # Load image
        image = cv2.imread(full_path)
        if image is None:
            # Handle missing images gracefully (though metadata check should prevent this)
            # Return a black image or raise error. Here we return black image.
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply transforms
        if self.transforms:
            augmented = self.transforms(image=image)
            image = augmented["image"]

        if self.is_test:
            return image, 0  # Dummy label
        else:
            label = self.labels[idx]
            return image, torch.tensor(label, dtype=torch.long)


class BalanceClassSampler(Sampler):
    """
    Abstractions for P-K sampling:
    P = classes_per_batch
    K = samples_per_class

    Ensures that each batch contains P unique classes, and K samples for each of those classes.
    """

    def __init__(self, labels, classes_per_batch, samples_per_class):
        self.labels = np.array(labels)
        self.classes_per_batch = classes_per_batch
        self.samples_per_class = samples_per_class
        self.batch_size = self.classes_per_batch * self.samples_per_class

        # Organize indices by class
        self.classes = np.unique(self.labels)
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}

        # Calculate number of batches
        # We want to cover roughly the dataset size, but strictly adhering to P-K structure
        self.n_samples = len(self.labels)
        self.n_batches = int(self.n_samples / self.batch_size)

    def __iter__(self):
        count = 0
        while count < self.n_batches:
            # 1. Select P classes randomly
            selected_classes = np.random.choice(
                self.classes, self.classes_per_batch, replace=False
            )

            batch_indices = []

            for c in selected_classes:
                # 2. Select K samples for each class
                # If a class has fewer than K samples, we sample with replacement
                indices = self.class_indices[c]
                if len(indices) >= self.samples_per_class:
                    selected_indices = np.random.choice(
                        indices, self.samples_per_class, replace=False
                    )
                else:
                    selected_indices = np.random.choice(
                        indices, self.samples_per_class, replace=True
                    )

                batch_indices.extend(selected_indices)

            # Shuffle the batch indices so classes are mixed within the batch
            np.random.shuffle(batch_indices)

            yield from batch_indices
            count += 1

    def __len__(self):
        return self.n_batches * self.batch_size

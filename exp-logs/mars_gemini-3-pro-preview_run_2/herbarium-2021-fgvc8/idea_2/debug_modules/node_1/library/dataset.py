import os
import cv2
import torch
import numpy as np
import pandas as pd
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from library.config import Config
from library.taxonomy import TaxonomyManager


def get_transforms(mode="train", image_size=224):
    """
    Returns the Albumentations transformation pipeline.

    Args:
        mode (str): 'train', 'val', or 'test'.
        image_size (int): Target image size.

    Returns:
        A.Compose: The composition of transforms.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(height=image_size, width=image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


class HerbariumDataset(Dataset):
    """
    Dataset class for the Herbarium 2021 competition.
    Handles loading images and hierarchical labels (Species, Family, Order).
    """

    def __init__(self, mode="train", transform=None, debug=False):
        """
        Args:
            mode (str): One of 'train', 'val', 'test'.
            transform (A.Compose): Albumentations transforms.
            debug (bool): If True, limits dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform
        self.input_dir = Config.INPUT_DIR

        # Load metadata based on mode
        if self.mode == "train":
            self.df = pd.read_csv(Config.TRAIN_CSV)
        elif self.mode == "val":
            self.df = pd.read_csv(Config.VAL_CSV)
        elif self.mode == "test":
            self.df = pd.read_csv(Config.TEST_CSV)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # Debugging: Limit dataset size
        if debug or Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE].reset_index(drop=True)

        # Taxonomy Mapping (only for train/val)
        self.taxonomy_map = None
        if self.mode in ["train", "val"]:
            self.tm = TaxonomyManager()
            self.taxonomy_map = self.tm.get_mappings()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct full image path
        # metadata file_path is relative to input dir (e.g., "train/images/...")
        image_path = os.path.join(self.input_dir, row["file_path"])

        # Load Image
        image = cv2.imread(image_path)
        if image is None:
            # Fallback for missing images (should ideally not happen after verification)
            # Create a black image of default size
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Apply Transforms
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        else:
            # Fallback to simple tensor conversion if no transform provided
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        # Return logic based on mode
        if self.mode == "test":
            image_id = row["image_id"]
            return image, image_id
        else:
            species_id = int(row["category_id"])

            # Retrieve auxiliary labels
            if self.taxonomy_map and species_id in self.taxonomy_map:
                aux = self.taxonomy_map[species_id]
                family_id = int(aux["family_id"])
                order_id = int(aux["order_id"])
            else:
                # Should not happen if taxonomy is consistent
                family_id = -1
                order_id = -1

            return image, species_id, family_id, order_id


class CutMixCollator:
    """
    Collator that applies CutMix regularization to a batch of data.
    """

    def __init__(self, alpha=1.0, p=0.5):
        """
        Args:
            alpha (float): Parameter for Beta distribution.
            p (float): Probability of applying CutMix.
        """
        self.alpha = alpha
        self.p = p

    def __call__(self, batch):
        """
        Args:
            batch: List of tuples (image, species, family, order)

        Returns:
            images (Tensor): Mixed images [B, C, H, W]
            targets (dict): Dictionary containing:
                - 'species': (target_a, target_b)
                - 'family': (target_a, target_b)
                - 'order': (target_a, target_b)
                - 'lam': Mixing coefficient lambda
        """
        # Unpack batch
        images, species, families, orders = zip(*batch)

        # Stack to tensors
        images = torch.stack(images)
        species = torch.tensor(species, dtype=torch.long)
        families = torch.tensor(families, dtype=torch.long)
        orders = torch.tensor(orders, dtype=torch.long)

        batch_size = images.size(0)

        # Decide whether to apply CutMix
        if np.random.rand() < self.p:
            # Generate mixing parameters
            lam = np.random.beta(self.alpha, self.alpha)
            rand_index = torch.randperm(batch_size)

            # Create bounding box
            bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)

            # Apply CutMix to images
            images[:, :, bby1:bby2, bbx1:bbx2] = images[
                rand_index, :, bby1:bby2, bbx1:bbx2
            ]

            # Adjust lambda to match exact pixel ratio
            lam = 1 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2])
            )

            # Prepare targets
            targets = {
                "species": (species, species[rand_index]),
                "family": (families, families[rand_index]),
                "order": (orders, orders[rand_index]),
                "lam": lam,
            }
        else:
            # No CutMix applied
            targets = {
                "species": (species, species),
                "family": (families, families),
                "order": (orders, orders),
                "lam": 1.0,
            }

        return images, targets

    @staticmethod
    def rand_bbox(size, lam):
        """
        Generates a random bounding box for CutMix.

        Args:
            size: Tensor size [B, C, H, W]
            lam: Lambda value from Beta distribution

        Returns:
            bbx1, bby1, bbx2, bby2
        """
        W = size[3]
        H = size[2]
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        # Uniform center
        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

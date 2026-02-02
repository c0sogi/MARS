import os
import json
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from library.utils import CFG
from library.data_processing import read_tiff, read_tiff_region


def get_transforms(data):
    """
    Returns the albumentations transformation pipeline.

    Args:
        data (str): 'train', 'valid', or 'test'.

    Returns:
        A.Compose: Composed transformations.
    """
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.RandomGamma(gamma_limit=(80, 120), p=1),
                        A.HueSaturationValue(
                            hue_shift_limit=20,
                            sat_shift_limit=30,
                            val_shift_limit=20,
                            p=1,
                        ),
                        A.RGBShift(
                            r_shift_limit=25, g_shift_limit=25, b_shift_limit=25, p=1
                        ),
                    ],
                    p=0.5,
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.OpticalDistortion(
                            distort_limit=0.05, shift_limit=0.05, p=1.0
                        ),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data == "valid" or data == "test":
        return A.Compose(
            [
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    else:
        raise ValueError(f"Unknown data mode: {data}")


class HuBMAPDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        """
        Args:
            df (pd.DataFrame): Dataframe containing tile information.
            transforms (A.Compose): Albumentations transforms.
            mode (str): 'train', 'valid', or 'test'.
        """
        self.df = df
        self.transforms = transforms
        self.mode = mode

        # Cache polygons for training/validation to speed up on-the-fly mask generation
        self.polygons = {}
        if self.mode in ["train", "valid"]:
            self._cache_polygons()

    def _cache_polygons(self):
        """
        Loads and caches glomerulus polygons for all images in the dataset.
        """
        unique_ids = self.df["id"].unique()
        for img_id in unique_ids:
            # Get the JSON path for this image
            subset = self.df[self.df["id"] == img_id]
            if subset.empty:
                continue

            # Take the first row's json path
            json_rel_path = subset.iloc[0]["json_path"]

            # Handle missing or NaN paths
            if pd.isna(json_rel_path):
                self.polygons[img_id] = []
                continue

            json_path = os.path.join(CFG.input_root, json_rel_path)
            if not os.path.exists(json_path):
                self.polygons[img_id] = []
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)

                polys = []
                # Parse GeoJSON-like structure
                for feature in data:
                    geom = feature.get("geometry", {})
                    coords = geom.get("coordinates", [])
                    # Coordinates are typically [[[x,y],...]]
                    for poly_coords in coords:
                        pts = np.array(poly_coords, dtype=np.int32)
                        polys.append(pts)

                self.polygons[img_id] = polys
            except Exception as e:
                print(f"Error parsing JSON for {img_id}: {e}")
                self.polygons[img_id] = []

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]

        # Paths
        img_path = os.path.join(CFG.input_root, row["image_path"])

        # Coordinates
        x, y, w, h = row["x"], row["y"], row["w"], row["h"]

        # Load Image Tile
        # Use context manager to ensure file handle is closed
        try:
            with read_tiff(img_path) as src:
                image = read_tiff_region(src, x, y, w, h)
        except Exception as e:
            # Fallback for robustness
            print(f"Error reading image {img_id} at {x},{y}: {e}")
            image = np.zeros((h, w, 3), dtype=np.uint8)

        # Ensure 3 channels (RGB)
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        elif image.shape[2] > 3:
            image = image[:, :, :3]

        mask = None
        if self.mode in ["train", "valid"]:
            # Generate Mask
            mask = np.zeros((h, w), dtype=np.float32)
            polys = self.polygons.get(img_id, [])

            if polys:
                # Define tile bounding box
                tile_box = [x, y, x + w, y + h]

                valid_polys = []
                for poly in polys:
                    # Quick bounding box check
                    p_min = poly.min(axis=0)
                    p_max = poly.max(axis=0)

                    # Check if polygon intersects with tile
                    if (
                        p_max[0] < tile_box[0]
                        or p_min[0] > tile_box[2]
                        or p_max[1] < tile_box[1]
                        or p_min[1] > tile_box[3]
                    ):
                        continue

                    # Shift coordinates relative to tile
                    shifted_poly = poly - np.array([x, y])
                    valid_polys.append(shifted_poly)

                if valid_polys:
                    cv2.fillPoly(mask, valid_polys, 1.0)

        # Apply Augmentations
        if self.transforms:
            if mask is not None:
                augmented = self.transforms(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transforms(image=image)
                image = augmented["image"]

        # Prepare Output
        result = {"image": image, "id": img_id, "x": x, "y": y, "h": h, "w": w}

        if mask is not None:
            # Ensure mask is (C, H, W) -> (1, H, W)
            # ToTensorV2 usually returns (H, W) for single channel mask
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            result["mask"] = mask

        return result

import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import set_seed


class InkDataset(Dataset):
    def __init__(self, df, mode="train", load_cached_data=True):
        """
        Dataset class for Siamese Multi-View SegFormer.

        Args:
            df (pd.DataFrame): Metadata dataframe.
            mode (str): 'train', 'val', or 'test'.
            load_cached_data (bool): Whether to use disk caching for volumes.
        """
        self.mode = mode
        self.load_cached_data = load_cached_data

        # Determine global Z-range required for all views
        # We need the min start of the lowest view and max end of the highest view
        # View starts: [16, 20, 24]
        # Channel offsets: [(0,12), (6,18), (12,24)]
        # Min Z = min(starts) + min(start_offsets) = 16 + 0 = 16
        # Max Z = max(starts) + max(end_offsets) = 24 + 24 = 48
        self.z_min_global = min(Config.VIEW_START_INDICES) + min(
            s for s, e in Config.CHANNEL_OFFSETS
        )
        self.z_max_global = max(Config.VIEW_START_INDICES) + max(
            e for s, e in Config.CHANNEL_OFFSETS
        )

        # If test mode, we need to expand the fragment-level metadata into tile-level metadata
        if self.mode == "test":
            self.df = self._expand_test_dataframe(df)
        else:
            self.df = df.reset_index(drop=True)

        # Load volumes into memory (or cache)
        self.volumes = {}
        self.masks = {}
        self._preload_fragments()

        # Define Augmentations (Geometric only, consistent across views)
        if self.mode == "train":
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    ToTensorV2(transpose_mask=True),
                ],
                additional_targets={"view_2": "image", "view_3": "image"},
            )
        else:
            self.transform = A.Compose(
                [ToTensorV2(transpose_mask=True)],
                additional_targets={"view_2": "image", "view_3": "image"},
            )

    def _preload_fragments(self):
        """
        Loads necessary 3D volumes and 2D masks for all fragments in the dataframe.
        Uses caching mechanism to store processed numpy arrays.
        """
        unique_fragments = self.df["fragment_id"].unique()

        for frag_id in unique_fragments:
            # Get paths from the first occurrence in dataframe
            row = self.df[self.df["fragment_id"] == frag_id].iloc[0]

            # 1. Load Mask
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(f"Mask not found at {mask_path}")
            # Binarize mask (0 or 255) -> (0 or 1)
            self.masks[frag_id] = (mask > 0).astype(np.uint8)

            # 2. Load Volume (Cached)
            volume_rel_path = row["volume_path"]
            self.volumes[frag_id] = self._load_volume_cached(frag_id, volume_rel_path)

    def _load_volume_cached(self, frag_id, volume_rel_path):
        """
        Loads the specific Z-slice range of a fragment volume.
        Checks disk cache first.
        """
        cache_filename = (
            f"frag_{frag_id}_slices_{self.z_min_global}_{self.z_max_global}.npy"
        )
        cache_path = os.path.join(Config.WORKING_DIR, cache_filename)

        # Try loading from cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                # print(f"Loaded fragment {frag_id} from cache: {cache_path}")
                return volume
            except Exception as e:
                print(f"Failed to load cache for {frag_id}: {e}. Recomputing.")

        # Load from TIFFs
        full_vol_dir = os.path.join(Config.INPUT_DIR, volume_rel_path)
        slices = []

        # Iterate through the required Z range
        for z in range(self.z_min_global, self.z_max_global):
            slice_path = os.path.join(full_vol_dir, f"{z:02d}.tif")
            if not os.path.exists(slice_path):
                # Fallback or error? For this competition, data should exist.
                # We pad with zeros if missing to maintain shape.
                # Assuming first slice exists to get shape.
                # If 00.tif exists we can get shape, but here we are deep in stack.
                # Let's assume strict existence.
                raise FileNotFoundError(f"Slice {slice_path} missing.")

            img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
            slices.append(img)

        volume = np.stack(slices, axis=0)  # Shape: (D, H, W)

        # Save to cache
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        np.save(cache_path, volume)
        # print(f"Saved fragment {frag_id} to cache: {cache_path}")

        return volume

    def _expand_test_dataframe(self, df):
        """
        Expands a fragment-level test dataframe into a tile-level dataframe.
        """
        expanded_rows = []
        for _, row in df.iterrows():
            frag_id = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Read mask to get dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape

            # Generate tiles
            # We use the same stride and tile size as training
            for y in range(0, h, Config.STRIDE):
                for x in range(0, w, Config.STRIDE):
                    # For test, we process everything, even if it goes out of bounds (we'll pad later)
                    expanded_rows.append(
                        {
                            "fragment_id": frag_id,
                            "x": x,
                            "y": y,
                            "width": Config.TILE_SIZE,
                            "height": Config.TILE_SIZE,
                            "mask_path": row["mask_path"],
                            "volume_path": row["volume_path"],
                            # No label path for test
                        }
                    )

        return pd.DataFrame(expanded_rows)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = row["fragment_id"]
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        # Retrieve full fragment data
        # volume shape: (Total_Z_Range, H_frag, W_frag)
        # mask shape: (H_frag, W_frag)
        frag_volume = self.volumes[frag_id]
        frag_mask = self.masks[frag_id]

        frag_h, frag_w = frag_mask.shape

        # --- Cropping ---
        # Calculate crop coordinates with padding if necessary
        # We need to handle edge cases where x+w > frag_w

        pad_h = max(0, (y + h) - frag_h)
        pad_w = max(0, (x + w) - frag_w)

        # Crop limits
        y_end = min(y + h, frag_h)
        x_end = min(x + w, frag_w)

        # Crop Volume: (D, h_crop, w_crop)
        crop_volume = frag_volume[:, y:y_end, x:x_end]

        # Crop Mask: (h_crop, w_crop)
        # For training, we load the label. For test, we might just use the valid mask.
        if self.mode in ["train", "val"]:
            # Load label on the fly or cache?
            # Labels are small PNGs, load on fly is okay, but we need the path.
            # The metadata has 'label_path'.
            label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
            full_label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            crop_label = full_label[y:y_end, x:x_end]
            crop_label = (crop_label > 0).astype(np.float32)
        else:
            # Dummy label for test
            crop_label = np.zeros((y_end - y, x_end - x), dtype=np.float32)

        # Apply Padding if we are at the edge
        if pad_h > 0 or pad_w > 0:
            crop_volume = np.pad(
                crop_volume,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )
            crop_label = np.pad(
                crop_label, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0
            )

        # --- Generate Siamese Views ---
        # self.z_min_global is the index 0 of crop_volume
        # We need to extract 3 views based on Config

        views = {}

        for i, view_start_z in enumerate(Config.VIEW_START_INDICES):
            # Calculate local index in the cropped volume
            # Local 0 corresponds to self.z_min_global
            local_start_z = view_start_z - self.z_min_global

            channels = []
            for ch_start_offset, ch_end_offset in Config.CHANNEL_OFFSETS:
                # Determine slice range for this channel
                s_start = local_start_z + ch_start_offset
                s_end = local_start_z + ch_end_offset

                # Extract slab
                slab = crop_volume[s_start:s_end, :, :]

                # MIP (Maximum Intensity Projection)
                if slab.shape[0] > 0:
                    mip = np.max(slab, axis=0)
                else:
                    mip = np.zeros(
                        (Config.TILE_SIZE, Config.TILE_SIZE), dtype=crop_volume.dtype
                    )

                channels.append(mip)

            # Stack channels -> (H, W, 3)
            view_img = np.stack(channels, axis=-1)

            # Normalize
            view_img = (view_img.astype(np.float32) - Config.PIXEL_MIN) / (
                Config.PIXEL_MAX - Config.PIXEL_MIN
            )
            view_img = np.clip(view_img, 0, 1)

            views[f"view_{i+1}"] = view_img

        # --- Augmentation ---
        # We need to augment all views and the mask identically
        # Albumentations handles this via additional_targets

        transformed = self.transform(
            image=views["view_1"],
            view_2=views["view_2"],
            view_3=views["view_3"],
            mask=crop_label,
        )

        # Prepare Output
        # Transpose is handled by ToTensorV2 (H,W,C -> C,H,W)
        inputs = {
            "view_1": transformed["image"],
            "view_2": transformed["view_2"],
            "view_3": transformed["view_3"],
        }

        # Mask shape from ToTensorV2 is usually (H, W) or (1, H, W) depending on setup
        # We want (1, H, W) float tensor
        target = transformed["mask"].unsqueeze(0).float()

        return inputs, target

import os
import cv2
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from library.config import Config


class InkDataset(Dataset):
    """
    Dataset class for Vesuvius Ink Detection.

    Implements:
    - 3D Volume Loading with Disk Caching (.npy)
    - "Overlapping Thick Slab" Projection (MIP)
    - Constrained Dynamic Sampling (Random Z-shift for training)
    - Deterministic Z-Scanning support (Fixed Z for inference)
    - Geometric Augmentations
    """

    def __init__(self, mode, z_start=None, load_cached_data=True):
        """
        Args:
            mode (str): 'train', 'val', or 'test'.
            z_start (int, optional): Fixed start index for Z-slice. Used for validation/inference.
                                     If None and mode is 'train', random sampling is used.
            load_cached_data (bool): Whether to load pre-processed volumes from disk cache.
        """
        self.mode = mode
        self.z_start = z_start
        self.load_cached_data = load_cached_data

        # In-memory cache to avoid reloading npy files for every patch
        self.volume_cache = {}
        self.mask_cache = {}

        # Load Metadata
        if self.mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif self.mode == "val":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
        elif self.mode == "test":
            self.df = self._generate_test_patches()
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def _generate_test_patches(self):
        """
        Generates patch metadata for test fragments by tiling the whole image.
        """
        test_meta = pd.read_csv(Config.TEST_METADATA_PATH)
        patches = []

        for _, row in test_meta.iterrows():
            frag_id = row["fragment_id"]
            mask_path = row["mask_path"]
            volume_path = row["volume_path"]

            # Load mask to get dimensions
            full_mask_path = os.path.join(Config.INPUT_DIR, mask_path)
            mask = cv2.imread(full_mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            h, w = mask.shape

            # Generate non-overlapping tiles (stride = tile_size)
            # We pad/clip at inference time usually, but here we strictly tile
            # based on top-left.
            for y in range(0, h, Config.TILE_SIZE):
                for x in range(0, w, Config.TILE_SIZE):
                    # Ensure patch is within bounds or handle partials
                    # For simplicity and standard submission, we usually pad
                    # or just take valid crops. Here we list all starts.
                    # The __getitem__ handles boundary clipping/padding if needed.
                    patches.append(
                        {
                            "fragment_id": frag_id,
                            "x": x,
                            "y": y,
                            "width": Config.TILE_SIZE,
                            "height": Config.TILE_SIZE,
                            "mask_path": mask_path,
                            "volume_path": volume_path,
                            # Labels are not available for test
                            "label_path": None,
                        }
                    )

        return pd.DataFrame(patches)

    def _load_volume(self, fragment_id, volume_rel_path):
        """
        Loads the 3D volume for a fragment.
        Implements caching logic: checks for .npy in working dir, else loads TIFFs.
        """
        # Return in-memory if available
        if fragment_id in self.volume_cache:
            return self.volume_cache[fragment_id]

        # Define cache path
        cache_filename = f"frag_{fragment_id}_volume.npy"
        cache_path = os.path.join(Config.CACHE_DIR, cache_filename)

        # 1. Try to load from disk cache
        if self.load_cached_data and os.path.exists(cache_path):
            try:
                volume = np.load(cache_path)
                self.volume_cache[fragment_id] = volume
                return volume
            except Exception as e:
                print(
                    f"Failed to load cached volume {cache_path}: {e}. Reloading from source."
                )

        # 2. Load from source TIFFs
        # We load a safe range of slices to cover all potential Z-shifts.
        # Min needed: Config.TRAIN_Z_MIN
        # Max needed: Config.TRAIN_Z_MAX + Config.SLAB_DEPTH
        # To be safe and support inference (which might use slightly different Zs),
        # we load a generous range or the whole set if small enough.
        # Given 65 slices per fragment, we can load indices 0 to 64.

        full_vol_path = os.path.join(Config.INPUT_DIR, volume_rel_path)
        slices = []

        # We assume standard naming 00.tif, 01.tif ... 64.tif
        # We'll load indices 0 to 64 to be safe.
        for i in range(65):
            slice_name = f"{i:02d}.tif"
            slice_path = os.path.join(full_vol_path, slice_name)

            if os.path.exists(slice_path):
                img = cv2.imread(slice_path, cv2.IMREAD_GRAYSCALE)
                slices.append(img)
            else:
                # If file doesn't exist (e.g. fewer than 65), we stop or pad.
                # Assuming contiguous block from 0.
                break

        if not slices:
            raise FileNotFoundError(f"No slice files found in {full_vol_path}")

        volume = np.stack(slices, axis=0)  # Shape: (D, H, W)

        # 3. Save to cache
        os.makedirs(Config.CACHE_DIR, exist_ok=True)
        np.save(cache_path, volume)

        self.volume_cache[fragment_id] = volume
        return volume

    def _load_image(self, rel_path, grayscale=True):
        if rel_path is None:
            return None
        path = os.path.join(Config.INPUT_DIR, rel_path)
        if not os.path.exists(path):
            return None
        flags = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_UNCHANGED
        return cv2.imread(path, flags)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        frag_id = str(row["fragment_id"])
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        # 1. Load Volume
        volume = self._load_volume(frag_id, row["volume_path"])
        vol_d, vol_h, vol_w = volume.shape

        # 2. Determine Z-Start
        if self.mode == "train":
            # Constrained Dynamic Sampling
            z_start = np.random.randint(Config.TRAIN_Z_MIN, Config.TRAIN_Z_MAX + 1)
        else:
            # Deterministic Inference/Validation
            z_start = self.z_start if self.z_start is not None else Config.TRAIN_Z_MIN

        # 3. Extract Slab
        z_end = z_start + Config.SLAB_DEPTH

        # Boundary check for Z
        if z_end > vol_d:
            z_end = vol_d
            z_start = max(0, z_end - Config.SLAB_DEPTH)

        slab = volume[z_start:z_end, :, :]  # (D_slab, H_full, W_full)

        # 4. Crop Spatial Patch
        # Handle edge cases where x+w or y+h exceeds volume dimensions
        # We pad with zeros if necessary
        pad_h = max(0, (y + h) - vol_h)
        pad_w = max(0, (x + w) - vol_w)

        crop_y = y
        crop_x = x
        crop_h = h - pad_h
        crop_w = w - pad_w

        slab_patch = slab[:, crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        if pad_h > 0 or pad_w > 0:
            # Pad (D, H, W)
            slab_patch = np.pad(
                slab_patch,
                ((0, 0), (0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=0,
            )

        # 5. Process into Channels (MIP)
        # Split slab into chunks
        chunk_size = Config.SLICES_PER_CHANNEL
        channels = []

        for i in range(Config.NUM_CHANNELS):
            start = i * chunk_size
            end = start + chunk_size
            # Get chunk, handle case if slab is smaller than expected (though unlikely with fixed depth)
            chunk = slab_patch[start:end, :, :]
            if chunk.shape[0] > 0:
                mip = np.max(chunk, axis=0)
            else:
                mip = np.zeros(
                    (Config.TILE_SIZE, Config.TILE_SIZE), dtype=slab_patch.dtype
                )
            channels.append(mip)

        image = np.stack(channels, axis=0)  # (3, H, W)

        # 6. Normalize
        image = image.astype(np.float32) / 65535.0  # 16-bit to [0,1]

        # Standardize using ImageNet stats
        # image is (C, H, W)
        mean = np.array(Config.NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
        std = np.array(Config.NORM_STD, dtype=np.float32).reshape(3, 1, 1)
        image = (image - mean) / std

        # 7. Load Label / Mask
        label = None
        if self.mode != "test" and row["label_path"] is not None:
            # Load full label image if not cached
            # Note: For efficiency in training, we might want to cache these too,
            # but they are 2D PNGs, so cv2.imread is reasonably fast.
            full_label = self._load_image(row["label_path"], grayscale=True)
            if full_label is not None:
                label_patch = full_label[
                    crop_y : crop_y + crop_h, crop_x : crop_x + crop_w
                ]
                if pad_h > 0 or pad_w > 0:
                    label_patch = np.pad(
                        label_patch,
                        ((0, pad_h), (0, pad_w)),
                        mode="constant",
                        constant_values=0,
                    )

                label = (label_patch > 0).astype(np.float32)
                label = np.expand_dims(label, axis=0)  # (1, H, W)

        # 8. Augmentations (Train Only)
        if self.mode == "train" and label is not None:
            # Random Flip
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=2).copy()  # Flip W
                label = np.flip(label, axis=2).copy()
            if np.random.rand() < 0.5:
                image = np.flip(image, axis=1).copy()  # Flip H
                label = np.flip(label, axis=1).copy()

            # Random Rotate
            k = np.random.randint(0, 4)
            if k > 0:
                image = np.rot90(image, k, axes=(1, 2)).copy()
                label = np.rot90(label, k, axes=(1, 2)).copy()

        # Convert to tensors
        image_tensor = torch.from_numpy(image).float()

        result = {"image": image_tensor, "fragment_id": frag_id, "x": x, "y": y}

        if label is not None:
            result["label"] = torch.from_numpy(label).float()

        return result

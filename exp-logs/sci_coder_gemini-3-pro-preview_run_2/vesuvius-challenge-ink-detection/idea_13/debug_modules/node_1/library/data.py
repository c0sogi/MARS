import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed


class VesuviusDataset(Dataset):
    def __init__(self, mode="train", transform=None, debug=False):
        """
        Args:
            mode (str): 'train', 'validation', or 'test'.
            transform (bool): Whether to apply geometric augmentations.
            debug (bool): If True, limits the dataset size for debugging.
        """
        self.mode = mode
        self.transform = transform
        self.debug = debug

        # Load Metadata
        if mode == "train":
            self.df = pd.read_csv(Config.TRAIN_METADATA_PATH)
        elif mode == "validation":
            self.df = pd.read_csv(Config.VAL_METADATA_PATH)
        elif mode == "test":
            self.df = self._expand_test_metadata(Config.TEST_METADATA_PATH)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        if self.debug:
            self.df = self.df.head(Config.MAX_DEBUG_SAMPLES)

        # Cache Data in RAM
        self.fragments = {}
        self._cache_data()

    def _expand_test_metadata(self, csv_path):
        """
        Expands fragment-level test metadata into patch-level metadata
        using a sliding window approach.
        """
        df_frag = pd.read_csv(csv_path)
        patch_data = []

        for _, row in df_frag.iterrows():
            frag_id = row["fragment_id"]
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])

            # Read mask to get dimensions
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
            h, w = mask.shape

            # Generate sliding window coordinates
            # We use the same stride as training for consistency, or TILE_SIZE for non-overlapping
            # The competition metric allows overlapping, but non-overlapping is simpler for submission.
            # We will use TILE_SIZE as stride for test to ensure full coverage without duplicate voting logic complexity
            # unless we implement a voting mechanism. Standard approach: non-overlapping or simple crop.
            # Config.STRIDE is used.

            for y in range(0, h, Config.STRIDE):
                for x in range(0, w, Config.STRIDE):
                    # Adjust edge cases
                    y_end = min(y + Config.TILE_SIZE, h)
                    x_end = min(x + Config.TILE_SIZE, w)

                    # If we are at the edge, we might crop smaller or shift back.
                    # Shift back strategy: if (x + size) > w, set x = w - size
                    real_y = y
                    real_x = x

                    if y_end == h:
                        real_y = max(0, h - Config.TILE_SIZE)
                    if x_end == w:
                        real_x = max(0, w - Config.TILE_SIZE)

                    patch_data.append(
                        {
                            "fragment_id": frag_id,
                            "x": real_x,
                            "y": real_y,
                            "width": Config.TILE_SIZE,
                            "height": Config.TILE_SIZE,
                            "mask_path": row["mask_path"],
                            "volume_path": row["volume_path"],
                        }
                    )

        # Remove duplicates if any (caused by shift back logic on small images)
        df_patches = pd.DataFrame(patch_data).drop_duplicates(
            subset=["fragment_id", "x", "y"]
        )
        return df_patches

    def _cache_data(self):
        """
        Loads necessary volumes and masks into memory.
        """
        fragment_ids = self.df["fragment_id"].unique()

        # Determine the global Z-range required by all views
        # Views: High(16-40), Center(20-44), Low(24-48)
        # Union: 16 to 48.
        min_slice = min(v[0] for v in Config.SLICES_VIEWS.values())
        max_slice = max(v[1] for v in Config.SLICES_VIEWS.values())

        print(
            f"[{self.mode.upper()}] Caching fragments: {fragment_ids} | Slices: {min_slice}-{max_slice}"
        )

        for fid in fragment_ids:
            # Get paths from the first occurrence in dataframe
            row = self.df[self.df["fragment_id"] == fid].iloc[0]
            vol_dir = os.path.join(Config.INPUT_DIR, row["volume_path"])

            # 1. Load Volume Slices
            volume_stack = []
            # Range is exclusive in python, so max_slice is not included if we do range(min, max).
            # Config ranges are likely inclusive-exclusive or inclusive-inclusive?
            # Standard python slicing [start:end].
            # Let's assume Config tuples are (start, end) where end is exclusive for python range.
            # Wait, 24 slices per view. 40-16=24. So (16, 40) means indices 16..39.
            # Max index needed is 48 (exclusive) -> 47.

            for i in range(min_slice, max_slice):
                slice_path = os.path.join(vol_dir, f"{i:02d}.tif")
                if not os.path.exists(slice_path):
                    # Fallback or error
                    print(f"Warning: Slice {slice_path} not found. Padding with zeros.")
                    # We need the shape. Load mask to get shape.
                    mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
                    ref_shape = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).shape
                    img = np.zeros(ref_shape, dtype=np.uint16)
                else:
                    img = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
                volume_stack.append(img)

            # Stack -> (Depth, H, W)
            volume_3d = np.stack(volume_stack, axis=0)

            # 2. Load Label (Ink) if available
            label_img = None
            if "label_path" in row and pd.notna(row["label_path"]):
                label_path = os.path.join(Config.INPUT_DIR, row["label_path"])
                if os.path.exists(label_path):
                    label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
                    # Binarize
                    label_img = (label_img > 0).astype(np.float32)

            # 3. Load Mask (Valid Pixels)
            mask_img = None
            mask_path = os.path.join(Config.INPUT_DIR, row["mask_path"])
            if os.path.exists(mask_path):
                mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                mask_img = (mask_img > 0).astype(np.float32)

            self.fragments[str(fid)] = {
                "volume": volume_3d,
                "label": label_img,
                "mask": mask_img,
                "offset": min_slice,  # To map global slice index to local stack index
            }

    def _process_view(self, volume_crop, view_range_global, stack_offset):
        """
        Projects a sub-volume into a 3-channel tensor using Overlapping Thick Slab.
        """
        start_global, end_global = view_range_global

        # Map to local indices in the cached stack
        start_local = start_global - stack_offset
        end_local = end_global - stack_offset

        # Extract sub-volume for this view
        # Shape: (Depth_View, H, W)
        sub_vol = volume_crop[start_local:end_local, :, :]

        # Split into 3 chunks for 3 channels
        # Assuming depth is divisible by 3 (24 / 3 = 8)
        depth = sub_vol.shape[0]
        chunk_size = depth // 3

        channels = []
        for i in range(3):
            c_start = i * chunk_size
            c_end = (i + 1) * chunk_size
            chunk = sub_vol[c_start:c_end, :, :]
            # Mean projection
            proj = np.mean(chunk, axis=0)
            channels.append(proj)

        # Stack to (3, H, W)
        img = np.stack(channels, axis=0)

        # Normalize [0, 1]
        img = img.astype(np.float32) / 65535.0

        return img

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fid = str(row["fragment_id"])
        x, y = int(row["x"]), int(row["y"])
        w, h = int(row["width"]), int(row["height"])

        data = self.fragments[fid]
        vol_full = data["volume"]
        offset = data["offset"]

        # Crop Volume
        # vol_full is (D, H_full, W_full)
        # Crop: (D, h, w)
        vol_crop = vol_full[:, y : y + h, x : x + w]

        # Generate Views
        # High
        view_high = self._process_view(vol_crop, Config.SLICES_VIEWS["high"], offset)
        # Center
        view_center = self._process_view(
            vol_crop, Config.SLICES_VIEWS["center"], offset
        )
        # Low
        view_low = self._process_view(vol_crop, Config.SLICES_VIEWS["low"], offset)

        # Get Label and Mask Crop
        label = np.zeros((h, w), dtype=np.float32)
        if self.mode != "test" and data["label"] is not None:
            label = data["label"][y : y + h, x : x + w]

        # Augmentation
        if self.transform:
            # Random discrete augmentations
            # 1. Rotation (0, 90, 180, 270)
            k = np.random.randint(0, 4)
            # 2. Flip (Horizontal)
            do_flip = np.random.rand() > 0.5

            # Helper to apply
            def apply_aug(img_tensor, is_3d=True):
                # img_tensor: (C, H, W) if is_3d else (H, W)
                if is_3d:
                    # Rotate H, W axes (1, 2)
                    res = np.rot90(img_tensor, k, axes=(1, 2))
                    if do_flip:
                        res = np.flip(res, axis=2)  # Flip W
                else:
                    res = np.rot90(img_tensor, k)
                    if do_flip:
                        res = np.flip(res, axis=1)
                return res.copy()

            view_high = apply_aug(view_high, True)
            view_center = apply_aug(view_center, True)
            view_low = apply_aug(view_low, True)
            label = apply_aug(label, False)

        # Convert to Torch Tensors
        # Inputs: (3, H, W) -> FloatTensor
        t_high = torch.from_numpy(view_high).float()
        t_center = torch.from_numpy(view_center).float()
        t_low = torch.from_numpy(view_low).float()

        # Label: (1, H, W)
        t_label = torch.from_numpy(label).float().unsqueeze(0)

        if self.mode == "test":
            # For test, we might need metadata to reconstruct
            return t_high, t_center, t_low, torch.tensor([idx])
        else:
            return t_high, t_center, t_low, t_label


def get_loaders():
    set_seed(Config.SEED)

    train_ds = VesuviusDataset(mode="train", transform=True)
    val_ds = VesuviusDataset(mode="validation", transform=False)
    test_ds = VesuviusDataset(mode="test", transform=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

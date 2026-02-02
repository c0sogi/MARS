import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

from library.config import Config
from library.utils import load_image, rle_decode, set_seed


def process_metadata(csv_path, mode="train", load_cached_data=True):
    """
    Processes the metadata CSV to create a wide-format dataframe suitable for 2.5D training.
    Handles caching to parquet to save processing time.

    Args:
        csv_path (str): Path to the source metadata CSV.
        mode (str): 'train', 'val', or 'test'. Used for naming the cache file.
        load_cached_data (bool): Whether to attempt loading from cache.

    Returns:
        pd.DataFrame: Processed dataframe with one row per slice and neighbor paths.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"processed_{mode}_metadata.parquet")

    # 1. Try to load from cache
    # Cite debug_lesson_7: Skip cache for test to ensure dynamic discovery of hidden files
    if mode != "test" and load_cached_data and os.path.exists(cache_path):
        try:
            df = pd.read_parquet(cache_path)
            return df
        except Exception as e:
            print(f"Failed to load cache {cache_path}: {e}. Recomputing...")

    if mode == "test":
        # Cite debug_lesson_7: Dynamically scan input directory for test files
        data = []
        test_dir = os.path.join(Config.INPUT_DIR, "test")
        for root, dirs, files in os.walk(test_dir):
            for file in files:
                if file.endswith(".png"):
                    # Format: slice_{number}_{width}_{height}_{spacing_w}_{spacing_h}.png
                    parts = file.replace(".png", "").split("_")
                    if len(parts) < 6:
                        continue

                    slice_id = parts[1]
                    w = int(parts[2])
                    h = int(parts[3])
                    pw = float(parts[4])
                    ph = float(parts[5])

                    # Parent dir: caseXXX_dayYY
                    parent = os.path.basename(os.path.dirname(root))
                    try:
                        c_str, d_str = parent.split("_")
                        case_id = int(c_str.replace("case", ""))
                        day_id = int(d_str.replace("day", ""))
                    except ValueError:
                        continue

                    img_id = f"{c_str}_{d_str}_slice_{slice_id}"
                    rel_path = os.path.relpath(
                        os.path.join(root, file), Config.INPUT_DIR
                    )

                    data.append(
                        {
                            "id": img_id,
                            "case": case_id,
                            "day": day_id,
                            "slice": int(slice_id),
                            "file_path": rel_path,
                            "img_width": w,
                            "img_height": h,
                            "pixel_spacing_w": pw,
                            "pixel_spacing_h": ph,
                        }
                    )

        df_wide = pd.DataFrame(data)
        # Ensure all class columns exist (empty for test)
        for cls in Config.CLASS_LABELS:
            df_wide[cls] = ""

    else:
        # 2. Load raw metadata
        df_raw = pd.read_csv(csv_path)

        # 3. Pivot to wide format (One row per slice, columns for each class mask)
        # Common columns that define a slice
        index_cols = [
            "id",
            "case",
            "day",
            "slice",
            "file_path",
            "img_width",
            "img_height",
            "pixel_spacing_w",
            "pixel_spacing_h",
        ]

        # Identify the segmentation column (train/val have 'segmentation', test has 'predicted' or none)
        seg_col = "segmentation" if "segmentation" in df_raw.columns else "predicted"
        if seg_col not in df_raw.columns:
            # Fallback for test if no prediction column exists
            df_raw[seg_col] = ""

        # Pivot
        # We want columns: large_bowel, small_bowel, stomach containing the RLE
        df_wide = df_raw.pivot_table(
            index=index_cols, columns="class", values=seg_col, aggfunc="first"
        ).reset_index()

        # Ensure all class columns exist
        for cls in Config.CLASS_LABELS:
            if cls not in df_wide.columns:
                df_wide[cls] = ""

    # 4. Sort for 2.5D logic
    df_wide = df_wide.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # 5. Compute neighbors (2.5D Context)
    # Shift file paths
    df_wide["prev_file_path"] = df_wide["file_path"].shift(1)
    df_wide["next_file_path"] = df_wide["file_path"].shift(-1)

    # Logic to handle boundaries (start/end of scan)
    # Check if case/day matches the shifted row. If not, it's a boundary.

    # Previous Slice Boundary
    # If current case != prev case OR current day != prev day, then prev_path = curr_path
    same_context_prev = (df_wide["case"] == df_wide["case"].shift(1)) & (
        df_wide["day"] == df_wide["day"].shift(1)
    )
    df_wide.loc[~same_context_prev, "prev_file_path"] = df_wide.loc[
        ~same_context_prev, "file_path"
    ]

    # Next Slice Boundary
    same_context_next = (df_wide["case"] == df_wide["case"].shift(-1)) & (
        df_wide["day"] == df_wide["day"].shift(-1)
    )
    df_wide.loc[~same_context_next, "next_file_path"] = df_wide.loc[
        ~same_context_next, "file_path"
    ]

    # 6. Save to cache
    os.makedirs(Config.WORKING_DIR, exist_ok=True)
    df_wide.to_parquet(cache_path, index=False)

    return df_wide


class UWDataset(Dataset):
    def __init__(self, df, mode="train", transforms=None, sample_ratio=0.5):
        """
        Args:
            df (pd.DataFrame): Processed dataframe (wide format).
            mode (str): 'train', 'val', or 'test'.
            transforms (albumentations.Compose): Augmentation pipeline.
            sample_ratio (float): Ratio of negative samples to keep in training.
        """
        self.mode = mode
        self.transforms = transforms

        # Balanced Sampling for Training
        if mode == "train":
            # Calculate if a slice has any mask
            # Check if strings are not empty and not NaN
            has_mask = (
                (df["large_bowel"].notna() & (df["large_bowel"] != ""))
                | (df["small_bowel"].notna() & (df["small_bowel"] != ""))
                | (df["stomach"].notna() & (df["stomach"] != ""))
            )

            df_pos = df[has_mask].copy()
            df_neg = df[~has_mask].copy()

            # Subsample negatives
            if len(df_neg) > 0:
                df_neg = df_neg.sample(frac=sample_ratio, random_state=Config.SEED)

            self.df = (
                pd.concat([df_pos, df_neg])
                .sample(frac=1, random_state=Config.SEED)
                .reset_index(drop=True)
            )
        else:
            self.df = df

        # Debugging subset
        if Config.DEBUG:
            self.df = self.df.iloc[: Config.DEBUG_SAMPLE_SIZE]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---------------------------
        # 1. Load 2.5D Image Stack
        # ---------------------------
        # Paths
        paths = [row["prev_file_path"], row["file_path"], row["next_file_path"]]
        images = []

        for p in paths:
            # load_image returns (H, W) normalized [0, 1]
            img = load_image(p)
            images.append(img)

        # Stack to (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # ---------------------------
        # 2. Prepare Masks (Train/Val)
        # ---------------------------
        if self.mode != "test":
            masks = []
            shape = (row["img_height"], row["img_width"])

            for cls in Config.CLASS_LABELS:
                rle = row[cls]
                mask = rle_decode(rle, shape)
                masks.append(mask)

            # Stack to (H, W, 3) -> One channel per class
            mask_stack = np.stack(masks, axis=-1).astype(np.float32)

            # ---------------------------
            # 3. Augmentations
            # ---------------------------
            if self.transforms:
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]
            else:
                # Basic Resize if no transforms provided (fallback, though transforms usually handle resize)
                # Assuming transforms includes ToTensorV2, which handles HWC -> CHW
                # If manual resize needed:
                img_stack = cv2.resize(img_stack, (Config.IMG_SIZE, Config.IMG_SIZE))
                mask_stack = cv2.resize(
                    mask_stack,
                    (Config.IMG_SIZE, Config.IMG_SIZE),
                    interpolation=cv2.INTER_NEAREST,
                )

                # Convert to tensor manually
                img_stack = torch.from_numpy(img_stack).permute(2, 0, 1).float()
                mask_stack = torch.from_numpy(mask_stack).permute(2, 0, 1).float()

            return img_stack, mask_stack

        # ---------------------------
        # 4. Inference Mode (Test)
        # ---------------------------
        else:
            # Only resize image
            if self.transforms:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]
            else:
                img_stack = cv2.resize(img_stack, (Config.IMG_SIZE, Config.IMG_SIZE))
                img_stack = torch.from_numpy(img_stack).permute(2, 0, 1).float()

            # Return metadata needed for submission formatting
            original_size = torch.tensor([row["img_height"], row["img_width"]])
            return img_stack, row["id"], original_size


def get_transforms(mode="train"):
    """
    Returns Albumentations transforms for the given mode.
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                # ShiftScaleRotate is a bit heavy but good, sticking to lightweight as per idea
                ToTensorV2(transpose_mask=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
                ToTensorV2(transpose_mask=True),
            ]
        )

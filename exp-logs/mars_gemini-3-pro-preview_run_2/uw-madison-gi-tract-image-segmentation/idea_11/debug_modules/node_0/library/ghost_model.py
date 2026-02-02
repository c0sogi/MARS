import os
import gc
import cv2
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import ndimage
from library.config import Config
from library.utils import set_seed, rle_encode, rle_decode, dice_coef, hausdorff_3d

# ==========================================
# 1. Model Architecture: Ghost U-Net
# ==========================================


class GhostModule(nn.Module):
    def __init__(
        self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True
    ):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(
                inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False
            ),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(
                init_channels,
                new_channels,
                dw_size,
                1,
                dw_size // 2,
                groups=init_channels,
                bias=False,
            ),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.oup, :, :]


import math


class GhostBottleneck(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2]

        self.conv = nn.Sequential(
            # pw
            GhostModule(inp, hidden_dim, kernel_size=1, relu=True),
            # dw
            (
                nn.Conv2d(
                    hidden_dim,
                    hidden_dim,
                    kernel_size,
                    stride,
                    kernel_size // 2,
                    groups=hidden_dim,
                    bias=False,
                )
                if stride == 2
                else nn.Identity()
            ),
            nn.BatchNorm2d(hidden_dim) if stride == 2 else nn.Identity(),
            nn.ReLU(inplace=True) if stride == 2 else nn.Identity(),
            # Squeeze-and-Excite could be added here, omitted for speed/simplicity as per design
            # pw-linear
            GhostModule(hidden_dim, oup, kernel_size=1, relu=False),
        )

        if stride == 1 and inp == oup:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class GhostUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super(GhostUNet, self).__init__()

        # --- Encoder (GhostNet-like) ---
        # Stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, 2, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        # Stage 1
        self.stage1 = nn.Sequential(
            GhostBottleneck(16, 16, 16, 3, 1, False),
            GhostBottleneck(16, 48, 24, 3, 2, False),
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            GhostBottleneck(24, 72, 24, 3, 1, False),
            GhostBottleneck(24, 72, 40, 5, 2, False),
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            GhostBottleneck(40, 120, 40, 5, 1, False),
            GhostBottleneck(40, 240, 80, 3, 2, False),
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            GhostBottleneck(80, 200, 80, 3, 1, False),
            GhostBottleneck(80, 184, 80, 3, 1, False),
            GhostBottleneck(80, 184, 80, 3, 1, False),
            GhostBottleneck(80, 480, 112, 3, 1, False),
            GhostBottleneck(112, 672, 112, 3, 1, False),
            GhostBottleneck(112, 672, 160, 5, 2, False),
        )

        # Bridge
        self.bridge = GhostBottleneck(160, 960, 160, 5, 1, False)

        # --- Decoder (Ghost Modules) ---
        # Up 4
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = GhostModule(
            160 + 112, 112, kernel_size=3, relu=True
        )  # Skip from end of stage 4 (before stride 2)

        # Up 3
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = GhostModule(
            112 + 40, 64, kernel_size=3, relu=True
        )  # Skip from stage 2

        # Up 2
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = GhostModule(
            64 + 24, 32, kernel_size=3, relu=True
        )  # Skip from stage 1

        # Up 1
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = GhostModule(
            32 + 16, 16, kernel_size=3, relu=True
        )  # Skip from stem/conv1 output?
        # Actually conv1 output is stride 2.
        # Let's trace sizes:
        # Input: 256
        # Conv1: 128 (16ch) -> Skip 1
        # Stage1: 128->64 (24ch) -> Skip 2
        # Stage2: 64->32 (40ch) -> Skip 3
        # Stage3: 32->16 (80ch) -> Skip 4
        # Stage4: 16->8 (160ch)

        # Redefining Decoder to match skips
        self.d4_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # 8->16
        self.d4_conv = GhostModule(160 + 80, 80, kernel_size=3)

        self.d3_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # 16->32
        self.d3_conv = GhostModule(80 + 40, 40, kernel_size=3)

        self.d2_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # 32->64
        self.d2_conv = GhostModule(40 + 24, 24, kernel_size=3)

        self.d1_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # 64->128
        self.d1_conv = GhostModule(24 + 16, 16, kernel_size=3)

        self.final_up = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )  # 128->256
        self.final_conv = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.conv1(x)  # 128, 16
        x1 = self.stage1(x0)  # 64, 24
        x2 = self.stage2(x1)  # 32, 40
        x3 = self.stage3(x2)  # 16, 80
        x4 = self.stage4(x3)  # 8, 160

        b = self.bridge(x4)  # 8, 160

        # Decoder
        d4 = self.d4_up(b)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.d4_conv(d4)

        d3 = self.d3_up(d4)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.d3_conv(d3)

        d2 = self.d2_up(d3)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.d2_conv(d2)

        d1 = self.d1_up(d2)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.d1_conv(d1)

        out = self.final_up(d1)
        out = self.final_conv(out)

        return out


# ==========================================
# 2. Data Processing & Dataset
# ==========================================


def prepare_data(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, identifies 2.5D neighbors, and caches the result.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Ensure sorted by case, day, slice
    df = df.sort_values(["case", "day", "slice"]).reset_index(drop=True)

    # Create columns for prev/next slice paths
    # We will use indices to reference rows in the same dataframe to save space
    # or just store paths. Storing paths is safer.

    # Group by case+day to handle boundaries
    df["group_id"] = df["case"].astype(str) + "_" + df["day"].astype(str)

    # Dictionary mapping group_id to list of (slice_num, file_path)
    # This is faster than pandas shift operations on large groups

    # Add absolute paths
    df["abs_path"] = df["file_path"].apply(lambda x: os.path.join(Config.INPUT_DIR, x))

    # We need to link slice i to slice i-1 and i+1
    # Let's create a lookup dictionary
    # key: (case, day, slice), value: abs_path
    path_lookup = {}
    for idx, row in df.iterrows():
        path_lookup[(row["case"], row["day"], row["slice"])] = row["abs_path"]

    def get_neighbor_path(case, day, slice_num, offset):
        target_slice = slice_num + offset
        key = (case, day, target_slice)
        if key in path_lookup:
            return path_lookup[key]
        else:
            # Boundary condition: replicate current slice
            return path_lookup[(case, day, slice_num)]

    df["prev_path"] = df.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], -1), axis=1
    )
    df["next_path"] = df.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], 1), axis=1
    )

    # Cache
    df.to_parquet(cache_path, index=False)
    print(f"Data cached to {cache_path}")
    return df


class StomachDataset(Dataset):
    def __init__(self, df, transforms=None, mode="train"):
        self.df = df
        self.transforms = transforms
        self.mode = mode
        self.classes = Config.CLASSES

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load 2.5D images
        paths = [row["prev_path"], row["abs_path"], row["next_path"]]
        images = []
        for p in paths:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:  # Fallback
                img = np.zeros(Config.IMG_SIZE, dtype=np.uint16)

            # Resize
            img = cv2.resize(img, Config.IMG_SIZE, interpolation=cv2.INTER_LINEAR)

            # Normalize to 0-1
            img = img.astype(np.float32)
            max_val = img.max()
            if max_val > 0:
                img = img / max_val
            else:
                img = img / 65535.0  # Fallback for empty
            images.append(img)

        # Stack: (H, W, 3)
        img_stack = np.stack(images, axis=-1)

        # Load Mask (if train/val)
        mask = np.zeros((Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.float32)
        if self.mode in ["train", "val"]:
            # The dataframe is long format (one row per class? No, metadata is long format)
            # Wait, metadata from script has one row per class.
            # But we need one row per slice for the dataset to be efficient.
            # We must pivot the dataframe before creating the dataset or handle it here.
            # The provided metadata script output shows: case101_day20_slice_0001, large_bowel ...
            # This means 3 rows per slice.
            # WE NEED TO GROUP BY SLICE ID.

            # NOTE: The prepare_data function above processes the raw metadata.
            # If the input df is still long format, we have duplicates for slice images.
            # We should pivot it in prepare_data or handle it.
            # Let's assume prepare_data handles unique slices.
            pass

        # Correction: The metadata is long format. We need to restructure it for the dataset.
        # The dataset should iterate over unique SLICES, not unique rows in metadata.
        # I will handle this grouping in prepare_data.

        if self.mode in ["train", "val"]:
            # Expecting 'segmentation_large_bowel', etc. in row if pivoted
            # OR we pass the full subset and query it.
            # To be efficient, let's assume df passed to Dataset is UNIQUE slices
            # and has columns 'rle_large_bowel', 'rle_small_bowel', 'rle_stomach'.

            for i, cls in enumerate(self.classes):
                rle = row.get(f"segmentation_{cls}", "")
                if pd.notna(rle) and rle != "":
                    mask_cls = rle_decode(rle, (row["img_height"], row["img_width"]))
                    mask_cls = cv2.resize(
                        mask_cls, Config.IMG_SIZE, interpolation=cv2.INTER_NEAREST
                    )
                    mask[:, :, i] = mask_cls

        # Augmentation
        if self.transforms:
            # Albumentations expects (H, W, C)
            augmented = self.transforms(image=img_stack, mask=mask)
            img_stack = augmented["image"]
            mask = augmented["mask"]

        # Permute to (C, H, W)
        img_stack = np.transpose(img_stack, (2, 0, 1))
        mask = np.transpose(mask, (2, 0, 1))

        return torch.tensor(img_stack), torch.tensor(mask), row["id"]


def process_metadata_pivot(metadata_path, cache_name, load_cached_data=True):
    """
    Loads metadata, pivots to wide format (one row per slice), adds 2.5D paths.
    """
    cache_path = os.path.join(Config.WORKING_DIR, f"{cache_name}_pivoted.parquet")

    if load_cached_data and os.path.exists(cache_path):
        print(f"Loading cached pivoted data from {cache_path}")
        return pd.read_parquet(cache_path)

    print(f"Processing and pivoting data from {metadata_path}...")
    df = pd.read_csv(metadata_path)

    # Pivot to get one row per slice with columns for each class segmentation
    # Columns to keep constant per slice
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

    # Check if 'segmentation' exists (train/val) or 'predicted' (test)
    if "segmentation" in df.columns:
        df_pivot = df.pivot_table(
            index=index_cols, columns="class", values="segmentation", aggfunc="first"
        ).reset_index()
        # Rename columns
        df_pivot.rename(
            columns={c: f"segmentation_{c}" for c in Config.CLASSES}, inplace=True
        )
    else:
        # Test set might not have segmentation, just unique slices
        df_pivot = df[index_cols].drop_duplicates().reset_index(drop=True)

    # 2.5D Logic (Same as before but on pivoted df)
    df_pivot["abs_path"] = df_pivot["file_path"].apply(
        lambda x: os.path.join(Config.INPUT_DIR, x)
    )

    path_lookup = {}
    for idx, row in df_pivot.iterrows():
        path_lookup[(row["case"], row["day"], row["slice"])] = row["abs_path"]

    def get_neighbor_path(case, day, slice_num, offset):
        target_slice = slice_num + offset
        key = (case, day, target_slice)
        return path_lookup.get(key, path_lookup[(case, day, slice_num)])

    df_pivot["prev_path"] = df_pivot.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], -1), axis=1
    )
    df_pivot["next_path"] = df_pivot.apply(
        lambda r: get_neighbor_path(r["case"], r["day"], r["slice"], 1), axis=1
    )

    df_pivot.to_parquet(cache_path, index=False)
    print(f"Pivoted data cached to {cache_path}")
    return df_pivot


# ==========================================
# 3. Training & Validation
# ==========================================


def bce_dice_loss(y_pred, y_true):
    # y_pred is logits, apply sigmoid
    y_pred = torch.sigmoid(y_pred)

    bce = nn.BCELoss()(y_pred, y_true)

    smooth = 1e-5
    intersection = (y_pred * y_true).sum(dim=(2, 3))
    union = y_pred.sum(dim=(2, 3)) + y_true.sum(dim=(2, 3))
    dice = 2.0 * (intersection + smooth) / (union + smooth)
    dice_loss = 1.0 - dice.mean()

    return 0.5 * bce + 0.5 * dice_loss


def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, _ in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = bce_dice_loss(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, device):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device)
            masks = masks.cpu().numpy()

            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(np.float32)

            for i in range(len(preds)):
                d = dice_coef(masks[i], preds[i])
                dice_scores.append(d)

    return np.mean(dice_scores)


# ==========================================
# 4. Inference & Post-processing
# ==========================================


def post_process_3d(case_preds, shape_dict):
    """
    Applies 3D connected components to keep only the largest object per class.
    case_preds: dict of {slice_num: pred_mask_3ch}
    shape_dict: dict of {slice_num: (h, w)}
    """
    slices = sorted(case_preds.keys())
    if not slices:
        return {}

    # Stack into 3D volume: (D, H, W, C) -> (C, D, H, W)
    # Note: Images are resized to 256x256, need to resize back later?
    # Strategy: Process at 256x256, then resize back to original shape for RLE.

    vol_depth = len(slices)
    vol_h, vol_w = Config.IMG_SIZE
    volume = np.zeros((3, vol_depth, vol_h, vol_w), dtype=np.uint8)

    for z, s_idx in enumerate(slices):
        # Transpose (H, W, C) -> (C, H, W)
        p = case_preds[s_idx].transpose(2, 0, 1)
        volume[:, z, :, :] = p

    # Process each class
    final_rles = {}  # key: slice_num, val: [rle_c1, rle_c2, rle_c3]

    for c in range(3):
        class_vol = volume[c]
        if class_vol.sum() == 0:
            continue

        # Connected components
        labeled, num_features = ndimage.label(class_vol)
        if num_features > 1:
            # Find largest
            sizes = ndimage.sum(class_vol, labeled, range(num_features + 1))
            largest_label = sizes[1:].argmax() + 1  # 0 is background
            class_vol = (labeled == largest_label).astype(np.uint8)

        volume[c] = class_vol

    # Convert back to RLE per slice
    for z, s_idx in enumerate(slices):
        orig_h, orig_w = shape_dict[s_idx]
        rles = []
        for c in range(3):
            mask_slice = volume[c, z, :, :]
            # Resize to original
            if (vol_h, vol_w) != (orig_h, orig_w):
                mask_slice = cv2.resize(
                    mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

            rles.append(rle_encode(mask_slice))
        final_rles[s_idx] = rles

    return final_rles


def inference(model, test_loader, test_df, device):
    model.eval()

    # Store predictions: case_id -> {slice_num: mask}
    case_predictions = {}
    case_shapes = {}  # case_id -> {slice_num: (h, w)}

    print("Running inference...")
    with torch.no_grad():
        for images, _, ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            preds = (preds > 0.5).astype(np.uint8)

            for i, img_id in enumerate(ids):
                # Parse ID
                # id format: caseXXX_dayYY_slice_ZZZZ
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                if case_day not in case_predictions:
                    case_predictions[case_day] = {}
                    case_shapes[case_day] = {}

                # Transpose to (H, W, C) for storage
                case_predictions[case_day][slice_num] = preds[i].transpose(1, 2, 0)

                # Get original shape from dataframe
                row = test_df[test_df["id"] == img_id].iloc[0]
                case_shapes[case_day][slice_num] = (row["img_height"], row["img_width"])

    # Post-process and format for submission
    submission_rows = []

    print("Post-processing 3D volumes...")
    for case_day, slices_data in case_predictions.items():
        shapes = case_shapes[case_day]
        refined_rles = post_process_3d(slices_data, shapes)

        # Prepare rows
        # We need to ensure we output for every slice in the test set
        # The test_df might have multiple rows per slice (one per class) or one per slice
        # The submission format is: id, class, predicted

        for s_idx, rles in refined_rles.items():
            # Reconstruct ID
            # Need to zero-pad slice number to 4 digits?
            # Check metadata format. The ID in metadata is like 'case123_day20_slice_0001'
            slice_str = f"{s_idx:04d}"
            full_id = f"{case_day}_slice_{slice_str}"

            for i, cls in enumerate(Config.CLASSES):
                submission_rows.append(
                    {"id": full_id, "class": cls, "predicted": rles[i]}
                )

    # Convert to DF
    sub_df = pd.DataFrame(submission_rows)
    return sub_df


# ==========================================
# 5. Main Execution
# ==========================================


def run_training():
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    train_df = process_metadata_pivot(
        Config.TRAIN_METADATA_PATH, "train_pivoted", load_cached_data=True
    )
    val_df = process_metadata_pivot(
        Config.VAL_METADATA_PATH, "val_pivoted", load_cached_data=True
    )

    # Debug mode subsampling
    if Config.DEBUG:
        train_df = train_df.sample(
            frac=Config.DATA_FRACTION, random_state=Config.SEED
        ).reset_index(drop=True)
        val_df = val_df.sample(
            frac=Config.DATA_FRACTION, random_state=Config.SEED
        ).reset_index(drop=True)

    # 2. Datasets & Loaders
    train_dataset = StomachDataset(train_df, mode="train")
    val_dataset = StomachDataset(val_df, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    # 3. Model
    model = GhostUNet(in_channels=3, num_classes=3).to(device)

    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )

    # 4. Training Loop
    best_dice = 0.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_dice = validate(model, val_loader, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_PATH)
            print(f"New best model saved with Dice: {best_dice:.6f}")

    print("Training complete.")

    # 5. Inference
    print("Starting inference...")
    # Load best model
    model.load_state_dict(torch.load(Config.MODEL_PATH, map_location=device))

    test_df = process_metadata_pivot(
        Config.TEST_METADATA_PATH, "test_pivoted", load_cached_data=True
    )
    test_dataset = StomachDataset(test_df, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=Config.PIN_MEMORY,
    )

    sub_df = inference(model, test_loader, test_df, device)

    # Ensure all test IDs are present
    # The submission file must match sample_submission.csv format
    # We generated rows for all predicted slices.
    # We should merge with sample submission to ensure order and completeness.

    sample_sub = pd.read_csv(os.path.join(Config.INPUT_DIR, "sample_submission.csv"))
    # sample_sub keys: id, class, predicted (empty or example)

    # Merge on id and class
    final_sub = sample_sub[["id", "class"]].merge(
        sub_df, on=["id", "class"], how="left"
    )
    final_sub["predicted"] = final_sub["predicted"].fillna("")

    final_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm
import gc

from library.config import Config
from library.utils import (
    load_image,
    rle_encode,
    rle_decode,
    keep_largest_component_3d,
    dice_coef,
)

# =========================================================================
# 1. Model Architecture: 2.5D BiSeNet
# =========================================================================


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            in_chan,
            out_chan,
            kernel_size=ks,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    """
    Shallow branch to preserve spatial detail.
    Downsamples 1/8 via 3 layers of stride 2.
    """

    def __init__(self, in_channels, out_channels):
        super(SpatialPath, self).__init__()
        self.layer1 = ConvBNReLU(in_channels, 64, ks=7, stride=2, padding=3)
        self.layer2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.layer3 = ConvBNReLU(64, out_channels, ks=3, stride=2, padding=1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ContextPath(nn.Module):
    """
    Deep branch using MobileNetV2 to extract semantic context.
    """

    def __init__(self, in_channels):
        super(ContextPath, self).__init__()
        # Load pretrained MobileNetV2
        # Note: MobileNetV2 takes 3 channel input by default.
        # Our 2.5D input is 3 channels, so this aligns perfectly.
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        self.features = backbone.features

        # MobileNetV2 output channels at last layer is 1280
        self.out_channels = 1280

    def forward(self, x):
        # Forward pass through backbone
        feat = self.features(x)
        return feat


class FeatureFusionModule(nn.Module):
    """
    Fuses Spatial and Context paths with Channel Attention.
    """

    def __init__(self, spatial_chan, context_chan, out_chan):
        super(FeatureFusionModule, self).__init__()
        # Project context to match spatial dimensions if needed,
        # but here we concatenate then reduce.

        self.concat_conv = ConvBNReLU(
            spatial_chan + context_chan, out_chan, ks=3, stride=1, padding=1
        )

        # Channel Attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv2d(out_chan, out_chan // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chan // 4, out_chan, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, spatial, context):
        # Context is typically smaller (1/32), Spatial is (1/8)
        # Upsample context to match spatial
        context_up = F.interpolate(
            context, size=spatial.shape[2:], mode="bilinear", align_corners=True
        )

        # Concatenate
        feat = torch.cat([spatial, context_up], dim=1)
        feat = self.concat_conv(feat)

        # Attention
        atten = self.gap(feat)
        atten = self.conv_atten(atten)
        feat_atten = feat * atten

        return feat_atten + feat


class BiSeNet25D(nn.Module):
    def __init__(self, num_classes=3):
        super(BiSeNet25D, self).__init__()

        self.spatial_path = SpatialPath(in_channels=3, out_channels=128)
        self.context_path = ContextPath(in_channels=3)

        # Context path output is 1280, Spatial is 128
        self.ffm = FeatureFusionModule(
            spatial_chan=128, context_chan=1280, out_chan=256
        )

        # Heads
        self.main_head = nn.Sequential(
            ConvBNReLU(256, 128, ks=3, stride=1, padding=1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

        # Aux head on Context Path output
        self.aux_head = nn.Sequential(
            ConvBNReLU(1280, 128, ks=3, stride=1, padding=1),
            nn.Conv2d(128, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # x shape: (B, 3, H, W)

        # Paths
        sp_out = self.spatial_path(x)  # (B, 128, H/8, W/8)
        cx_out = self.context_path(x)  # (B, 1280, H/32, W/32)

        # Fusion
        feat_fuse = self.ffm(sp_out, cx_out)  # (B, 256, H/8, W/8)

        # Main Prediction
        main_out = self.main_head(feat_fuse)
        main_out = F.interpolate(
            main_out, size=x.shape[2:], mode="bilinear", align_corners=True
        )

        # Aux Prediction (Training only usually, but we return both)
        aux_out = self.aux_head(cx_out)
        aux_out = F.interpolate(
            aux_out, size=x.shape[2:], mode="bilinear", align_corners=True
        )

        return main_out, aux_out


# =========================================================================
# 2. Dataset with 2.5D Logic
# =========================================================================


class StomachIntestineDataset(Dataset):
    def __init__(self, df, phase="train", load_cached_data=True):
        self.phase = phase
        self.classes = Config.CLASS_LABELS
        self.img_size = Config.IMG_SIZE

        # Data Caching Logic
        cache_file = os.path.join(
            Config.WORKING_DIR, f"processed_{phase}_metadata.parquet"
        )

        if load_cached_data and os.path.exists(cache_file):
            print(f"Loading cached {phase} data from {cache_file}")
            self.df = pd.read_parquet(cache_file)
        else:
            print(f"Processing {phase} data...")
            # We need to build neighbor lookups
            # 1. Create a lookup dict: (case, day, slice) -> file_path
            # To avoid iterating slowly, we can just sort and use shift,
            # but cases might be interleaved in raw csv (though usually grouped).
            # A dictionary is safest.

            # Ensure we have unique rows per image for the lookup (train.csv has multiple rows per image)
            unique_imgs = df[
                ["case", "day", "slice", "file_path", "id"]
            ].drop_duplicates()

            # Create lookup key
            unique_imgs["lookup_key"] = list(
                zip(unique_imgs.case, unique_imgs.day, unique_imgs.slice)
            )
            path_map = dict(zip(unique_imgs["lookup_key"], unique_imgs["file_path"]))

            # 2. Add neighbor paths to the main dataframe
            # We process the unique images df first to find neighbors
            def get_neighbor_path(row, offset):
                key = (row.case, row.day, row.slice + offset)
                return path_map.get(key, None)  # None if out of bounds

            unique_imgs["prev_path"] = unique_imgs.apply(
                lambda r: get_neighbor_path(r, -1), axis=1
            )
            unique_imgs["next_path"] = unique_imgs.apply(
                lambda r: get_neighbor_path(r, 1), axis=1
            )

            # 3. Merge back to main df
            # If train/val, df has multiple rows per id (one per class).
            # If test, df might be one row per id or multiple.
            self.df = df.merge(
                unique_imgs[["id", "prev_path", "next_path"]], on="id", how="left"
            )

            # 4. Filter for sampling (Idea 9: All positives + 50% negatives)
            if phase == "train":
                # Check if any class has a mask for this image
                # Group by ID and check if any segmentation is not NaN
                # However, the df is long format.
                # Let's pivot or check efficiently.
                # We can just check 'segmentation' column availability.

                # Identify IDs that have at least one mask
                ids_with_mask = self.df[self.df["segmentation"].notna()]["id"].unique()

                # Split into pos and neg
                pos_df = self.df[self.df["id"].isin(ids_with_mask)]
                neg_df = self.df[~self.df["id"].isin(ids_with_mask)]

                # Subsample negatives
                if not neg_df.empty:
                    neg_ids = neg_df["id"].unique()
                    sample_size = int(len(neg_ids) * Config.NEGATIVE_SAMPLING_RATIO)
                    sampled_neg_ids = np.random.choice(
                        neg_ids, sample_size, replace=False
                    )
                    neg_df = neg_df[neg_df["id"].isin(sampled_neg_ids)]

                self.df = pd.concat([pos_df, neg_df]).reset_index(drop=True)

                # Shuffle
                self.df = self.df.sample(frac=1, random_state=Config.SEED).reset_index(
                    drop=True
                )

            # Save to cache
            os.makedirs(Config.WORKING_DIR, exist_ok=True)
            self.df.to_parquet(cache_file)
            print(f"Saved processed {phase} data to {cache_file}")

        # For efficient access, we want one item per Image ID, containing all class masks
        # Pivot the dataframe so we have columns: id, file_path, prev_path, next_path, rle_large_bowel, rle_small_bowel, rle_stomach
        if phase != "test":
            self.data = self.df.pivot_table(
                index=["id", "file_path", "prev_path", "next_path"],
                columns="class",
                values="segmentation",
                aggfunc="first",
            ).reset_index()
        else:
            # Test set structure might differ, usually one row per id/class in sample_submission
            # We just need unique images
            self.data = self.df[
                ["id", "file_path", "prev_path", "next_path"]
            ].drop_duplicates()

        # Debugging limit
        if Config.DEBUG:
            self.data = self.data.head(Config.DEBUG_SAMPLE_SIZE)
            print(f"DEBUG: Reduced dataset to {len(self.data)} samples")

    def __len__(self):
        return len(self.data)

    def _load_and_process_image(self, path):
        if path is None or pd.isna(path):
            return None
        img = load_image(path)
        # Resize
        img = cv2.resize(img, self.img_size, interpolation=cv2.INTER_LINEAR)
        return img

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # 1. Load 2.5D Stack
        curr_img = self._load_and_process_image(row["file_path"])
        prev_img = self._load_and_process_image(row["prev_path"])
        next_img = self._load_and_process_image(row["next_path"])

        # Handle boundaries (Replicate padding)
        if prev_img is None:
            prev_img = curr_img
        if next_img is None:
            next_img = curr_img

        # Stack: (H, W, 3)
        img_stack = np.stack([prev_img, curr_img, next_img], axis=-1)

        # Normalize [0, 1]
        # Min-max per stack
        mi, ma = img_stack.min(), img_stack.max()
        if ma > mi:
            img_stack = (img_stack - mi) / (ma - mi)
        else:
            img_stack = np.zeros_like(img_stack, dtype=np.float32)

        img_stack = img_stack.astype(np.float32)
        # To Tensor (C, H, W)
        img_tensor = torch.from_numpy(img_stack.transpose(2, 0, 1))

        # 2. Load Masks (if train/val)
        if self.phase != "test":
            masks = []
            for cls in self.classes:
                rle = row[cls] if cls in row else None
                if pd.isna(rle):
                    mask = np.zeros(self.img_size, dtype=np.float32)
                else:
                    # Decode requires original shape.
                    # We don't have original shape easily here without loading metadata again or parsing filename.
                    # However, load_image loads the original image, we can get shape from curr_img before resize?
                    # Optimization: We already resized curr_img.
                    # We need original shape to decode RLE correctly.
                    # Let's assume we can get it from the loaded image before resize,
                    # but we didn't keep it.
                    # Re-read original shape from file is slow.
                    # Alternative: Store original shape in cache.
                    # For now, let's assume standard shape or re-read (cached by OS).
                    # Actually, rle_decode needs the shape of the image *the RLE was created for*.
                    # The metadata has img_width, img_height. But we pivoted and lost it.
                    # Let's rely on the fact that we can get shape from loading the file again (fast enough)
                    # OR better, include width/height in pivot.
                    pass

            # Re-implementation for mask loading with shape awareness
            # We need original dimensions to decode RLE
            # Let's grab original shape from the raw image file since we load it anyway?
            # No, we loaded and resized immediately.
            # Let's peek at the file again or use a default if standard.
            # Most images are 266x266 or 360x310.
            # Let's load the original image simply to get shape for decoding.
            # This adds I/O overhead.
            # Better approach: The RLE decode function needs (H, W).
            # We can get this from the original image loading step.

            # Refined Flow:
            # Load original image -> get shape -> resize image -> decode mask using shape -> resize mask

            # Reload original for shape (cached by OS hopefully)
            # In _load_and_process_image, we can return shape.

            # Let's fix _load_and_process_image to return (img, original_shape)
            # But we call it 3 times.

            # Let's just load the current image raw once here.
            raw_curr = load_image(row["file_path"])
            h, w = raw_curr.shape[:2]

            mask_stack = []
            for cls in self.classes:
                rle = row[cls] if cls in row else None
                mask = rle_decode(rle, (h, w))
                mask = cv2.resize(mask, self.img_size, interpolation=cv2.INTER_NEAREST)
                mask_stack.append(mask)

            mask_tensor = torch.tensor(np.stack(mask_stack), dtype=torch.float32)

            return img_tensor, mask_tensor, row["id"]

        else:
            return img_tensor, row["id"]


# =========================================================================
# 3. Loss & Training Utils
# =========================================================================


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)

        pred_sigmoid = torch.sigmoid(pred)
        dice = dice_coef(
            target.detach().cpu().numpy(), pred_sigmoid.detach().cpu().numpy()
        )
        # We want to minimize (1 - Dice)
        dice_loss = 1 - dice

        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks, _ in tqdm(loader, desc="Training", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        main_out, aux_out = model(images)

        # Loss = Main + 0.1 * Aux
        loss_main = criterion(main_out, masks)
        loss_aux = criterion(aux_out, masks)
        loss = loss_main + Config.AUX_LOSS_WEIGHT * loss_aux

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    with torch.no_grad():
        for images, masks, _ in tqdm(loader, desc="Validation", leave=False):
            images = images.to(device)
            masks = masks.to(device)

            main_out, _ = model(images)
            loss = criterion(main_out, masks)
            running_loss += loss.item()

            # Metric: Dice
            preds = torch.sigmoid(main_out).cpu().numpy()
            targets = masks.cpu().numpy()

            # Threshold
            preds = (preds > 0.5).astype(np.float32)

            batch_dice = dice_coef(targets, preds)
            dice_scores.append(batch_dice)

    return running_loss / len(loader), np.mean(dice_scores)


# =========================================================================
# 4. Main Pipeline
# =========================================================================


def train_model():
    # Setup
    Config.setup()
    device = Config.DEVICE

    # Data
    print("Initializing Datasets...")
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    train_dataset = StomachIntestineDataset(train_df, phase="train")
    val_dataset = StomachIntestineDataset(val_df, phase="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    print(f"Initializing {Config.MODEL_NAME}...")
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES).to(device)

    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=1e-6
    )

    best_dice = 0.0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New Best Model Saved! (Dice: {best_dice:.6f})")

    print("Training Complete.")


def predict_and_submit():
    Config.setup()
    device = Config.DEVICE

    # Load Model
    model = BiSeNet25D(num_classes=Config.NUM_CLASSES).to(device)
    if not os.path.exists(Config.MODEL_SAVE_PATH):
        print("No trained model found. Please train first.")
        return

    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Data
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)
    test_dataset = StomachIntestineDataset(test_df, phase="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Store predictions: {case_day: {slice_idx: pred_volume_slice}}
    # We need to reconstruct 3D volumes for post-processing
    # Map: case_day -> (list of slice indices, list of predictions)
    volume_buffer = {}

    # Also need original dimensions to resize back
    # We can get this from test_df
    meta_map = test_df.set_index("id")[["img_width", "img_height"]].to_dict("index")

    print("Running Inference...")
    with torch.no_grad():
        for images, ids in tqdm(test_loader, desc="Inference"):
            images = images.to(device)
            outputs, _ = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()  # (B, 3, 256, 256)

            for i, img_id in enumerate(ids):
                # Parse case_day from id (caseXXX_dayYY_slice_ZZZZ)
                parts = img_id.split("_")
                case_day = f"{parts[0]}_{parts[1]}"
                slice_num = int(parts[3])

                if case_day not in volume_buffer:
                    volume_buffer[case_day] = []

                # Resize back to original
                orig_h = meta_map[img_id]["img_height"]
                orig_w = meta_map[img_id]["img_width"]

                # Transpose pred to (256, 256, 3) for resize then back
                p_trans = preds[i].transpose(1, 2, 0)
                p_resized = cv2.resize(
                    p_trans, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )

                # Binarize
                p_binary = (p_resized > 0.5).astype(np.uint8)  # (H, W, 3)

                volume_buffer[case_day].append((slice_num, img_id, p_binary))

    # Post-processing & Submission Generation
    print("Post-processing and Generating Submission...")
    results = []

    for case_day, slices_data in tqdm(volume_buffer.items(), desc="3D Processing"):
        # Sort by slice number
        slices_data.sort(key=lambda x: x[0])

        # Unpack
        slice_nums, img_ids, masks_list = zip(*slices_data)

        # Stack into volume: (Depth, H, W, 3)
        # Note: H and W might vary within a case? Usually not for the same scan.
        # Assuming consistent size within a scan.
        volume_4d = np.stack(masks_list, axis=0)  # (D, H, W, 3)

        # Process each class separately
        for cls_idx, cls_name in enumerate(Config.CLASS_LABELS):
            # Extract 3D volume for this class
            vol_cls = volume_4d[..., cls_idx]  # (D, H, W)

            # Keep largest connected component in 3D
            vol_cls_clean = keep_largest_component_3d(vol_cls)

            # Encode each slice
            for d, img_id in enumerate(img_ids):
                mask_slice = vol_cls_clean[d]
                rle = rle_encode(mask_slice)
                results.append({"id": img_id, "class": cls_name, "predicted": rle})

    # Save
    sub_df = pd.DataFrame(results)
    # Ensure column order
    sub_df = sub_df[["id", "class", "predicted"]]
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

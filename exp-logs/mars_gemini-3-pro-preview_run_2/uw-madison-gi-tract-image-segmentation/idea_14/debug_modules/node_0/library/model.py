import os
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label as scipy_label

from library.config import Config
from library.utils import load_image, rle_encode, calculate_dice, calculate_hausdorff_3d
from library.losses import BCETverskyLoss

# =============================================================================
# Model Architecture
# =============================================================================


class HRNetSegmentation(nn.Module):
    """
    2.5D HRNet-W32 for Segmentation.

    Uses HRNet-W32 backbone from timm.
    Input: 3 channels (Slice i-1, Slice i, Slice i+1).
    Output: 3 classes (Large Bowel, Small Bowel, Stomach).
    """

    def __init__(self, num_classes=3, pretrained=True):
        super(HRNetSegmentation, self).__init__()

        # Load HRNet-W32 backbone
        # features_only=True returns feature maps from different stages
        self.backbone = timm.create_model(
            Config.ARCH,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # HRNet-W32 typically outputs features at strides [4, 8, 16, 32]
        # with channels [32, 64, 128, 256]
        self.feature_channels = self.backbone.feature_info.channels()

        # Calculate total channels after concatenating upsampled features
        self.total_channels = sum(self.feature_channels)

        # Segmentation Head
        # 1. Projection/Mixing layer
        self.project = nn.Sequential(
            nn.Conv2d(
                self.total_channels, self.total_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.total_channels),
            nn.ReLU(inplace=True),
        )

        # 2. Final classification layer
        self.cls_head = nn.Conv2d(self.total_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # x shape: (B, 3, H, W)

        # Extract features
        features = self.backbone(x)
        # features is a list of tensors:
        # f0: stride 4, 32 ch
        # f1: stride 8, 64 ch
        # f2: stride 16, 128 ch
        # f3: stride 32, 256 ch

        # Upsample all features to the resolution of the first feature map (Stride 4)
        target_h, target_w = features[0].shape[2], features[0].shape[3]

        upsampled_features = [features[0]]
        for i in range(1, len(features)):
            upsampled_features.append(
                F.interpolate(
                    features[i],
                    size=(target_h, target_w),
                    mode="bilinear",
                    align_corners=True,
                )
            )

        # Concatenate features
        x = torch.cat(upsampled_features, dim=1)

        # Apply projection
        x = self.project(x)

        # Apply classification head
        x = self.cls_head(x)

        # Upsample to original input resolution (Stride 4 -> Stride 1)
        x = F.interpolate(x, scale_factor=4, mode="bilinear", align_corners=True)

        return x


# =============================================================================
# Dataset
# =============================================================================


class MedicalDataset(Dataset):
    def __init__(self, df, phase="train", transform=None):
        """
        Args:
            df (pd.DataFrame): Metadata dataframe.
            phase (str): 'train', 'val', or 'test'.
            transform (albumentations.Compose): Augmentations.
        """
        self.df = df
        self.phase = phase
        self.transform = transform

        # Create a lookup for (case, day, slice) -> file_path
        self.path_lookup = {}
        # Also lookup for ground truth masks if available
        self.mask_lookup = {}

        # Group by ID to handle multiple classes per slice
        # The input df is long format (one row per class per slice)
        # We need to pivot or group to get all classes for one image
        self.ids = self.df["id"].unique()

        # Pre-process lookup tables
        # We iterate through the dataframe to build the lookups
        # This might take a few seconds but saves time in __getitem__
        groups = self.df.groupby(["case", "day", "slice"])

        temp_data = []

        for (case, day, slc), group in groups:
            # All rows in group share the same file info
            row = group.iloc[0]
            file_path = row["file_path"]
            pixel_spacing = (row["pixel_spacing_h"], row["pixel_spacing_w"])  # h, w

            key = (case, day, slc)
            self.path_lookup[key] = {"path": file_path, "spacing": pixel_spacing}

            # Store masks if training/val
            if phase != "test":
                # Create mask array (H, W, 3)
                # Order: large_bowel, small_bowel, stomach
                # Note: We don't load the mask here, just store the RLEs
                rles = [None] * 3
                for _, r in group.iterrows():
                    cls_idx = Config.CLASSES.index(r["class"])
                    rles[cls_idx] = r["segmentation"]
                self.mask_lookup[key] = rles

            temp_data.append(
                {
                    "id": row["id"],
                    "case": case,
                    "day": day,
                    "slice": slc,
                    "orig_h": row["img_height"],
                    "orig_w": row["img_width"],
                }
            )

        self.meta_data = pd.DataFrame(temp_data)

    def __len__(self):
        if Config.SAMPLE_SIZE and Config.DEBUG:
            return min(len(self.meta_data), Config.SAMPLE_SIZE)
        return len(self.meta_data)

    def __getitem__(self, idx):
        row = self.meta_data.iloc[idx]
        case, day, slc = row["case"], row["day"], row["slice"]

        # 1. Load 2.5D Input (Slice i-1, i, i+1)
        imgs = []
        for d in [-1, 0, 1]:
            # Handle boundary conditions by clamping
            # We check if neighbor exists, if not use current slice
            neighbor_key = (case, day, slc + d)
            if neighbor_key in self.path_lookup:
                info = self.path_lookup[neighbor_key]
            else:
                info = self.path_lookup[(case, day, slc)]

            path = os.path.join(Config.INPUT_DIR, info["path"])
            img = load_image(path)
            imgs.append(img)

        # Stack to (H, W, 3)
        image = np.stack(imgs, axis=-1)

        # 2. Load Mask (if not test)
        mask = np.zeros(
            (row["orig_h"], row["orig_w"], Config.NUM_CLASSES), dtype=np.float32
        )
        if self.phase != "test":
            rles = self.mask_lookup[(case, day, slc)]
            for i, rle in enumerate(rles):
                if rle is not None and isinstance(rle, str):
                    mask[:, :, i] = rle_decode(rle, (row["orig_h"], row["orig_w"]))

        # 3. Physical Space Normalization
        # Resample to TARGET_SPACING (1.5mm)
        current_spacing = self.path_lookup[(case, day, slc)]["spacing"]  # (h, w)
        scale_h = current_spacing[0] / Config.TARGET_SPACING
        scale_w = current_spacing[1] / Config.TARGET_SPACING

        if scale_h != 1.0 or scale_w != 1.0:
            new_h = int(round(row["orig_h"] * scale_h))
            new_w = int(round(row["orig_w"] * scale_w))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            if self.phase != "test":
                mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # 4. Augmentation / Cropping
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        else:
            # For validation/test, we convert to tensor but keep variable size
            # (Batch size must be 1)
            image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            mask = torch.from_numpy(mask.transpose(2, 0, 1)).float()

        return {
            "image": image,
            "mask": mask,
            "id": row["id"],
            "orig_shape": np.array([row["orig_h"], row["orig_w"]]),
            "case": case,
            "day": day,
            "slice": slc,
        }


# =============================================================================
# Training & Inference Logic
# =============================================================================


def get_transforms(phase):
    if phase == "train":
        return A.Compose(
            [
                # Random Crop to fixed size
                A.PadIfNeeded(
                    min_height=Config.TRAIN_CROP_SIZE[0],
                    min_width=Config.TRAIN_CROP_SIZE[1],
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                ),
                A.RandomCrop(
                    height=Config.TRAIN_CROP_SIZE[0], width=Config.TRAIN_CROP_SIZE[1]
                ),
                # Spatial Augmentations
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=30, p=0.5),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
                A.GridDistortion(p=0.2),
                ToTensorV2(),
            ]
        )
    else:
        # Validation/Test: No cropping here, handled by sliding window or full image
        # We just need ToTensorV2 equivalent but we do it manually in dataset to handle variable sizes
        return None


def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Training", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []

    # For validation, we process one image at a time (batch_size=1)
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            # Handle padding for HRNet (needs dimensions divisible by 32)
            # We pad right and bottom
            h, w = images.shape[2], images.shape[3]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

            with autocast():
                outputs = model(images)

                # Crop back
                outputs = outputs[:, :, :h, :w]
                loss = criterion(outputs, masks)

            running_loss += loss.item() * images.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            targets = masks.cpu().numpy()

            for i in range(preds.shape[0]):
                dice_scores.append(calculate_dice(targets[i], preds[i]))

    return running_loss / len(loader.dataset), np.mean(dice_scores)


def train_model():
    print(f"Initializing Idea 14: HRNet with Physical Normalization...")
    device = Config.DEVICE

    # Load Metadata
    df_train = pd.read_csv(Config.TRAIN_METADATA_PATH)
    df_val = pd.read_csv(Config.VAL_METADATA_PATH)

    # Datasets
    train_dataset = MedicalDataset(
        df_train, phase="train", transform=get_transforms("train")
    )
    val_dataset = MedicalDataset(df_val, phase="val", transform=get_transforms("val"))

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )
    # Val loader must have batch_size=1 due to variable image sizes
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = HRNetSegmentation(num_classes=Config.NUM_CLASSES).to(device)

    # Optimizer & Loss
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
    )
    criterion = BCETverskyLoss(
        alpha=Config.TVERSKY_ALPHA,
        beta=Config.TVERSKY_BETA,
        smooth=Config.TVERSKY_SMOOTH,
        bce_weight=Config.WEIGHT_BCE,
        tversky_weight=Config.WEIGHT_TVERSKY,
    )
    scaler = GradScaler()

    best_dice = 0.0

    print("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device
        )
        val_loss, val_dice = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Dice: {val_dice:.6f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            print(f"New best model saved with Dice: {val_dice:.6f}")

    print("Training complete.")


def keep_largest_connected_component(mask):
    """
    Keeps only the largest connected component for each class slice-wise.
    Simple heuristic to reduce noise for Hausdorff distance.
    """
    # mask shape: (H, W)
    mask = mask.astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )

    if num_labels < 2:
        return mask

    # stats[:, 4] is area. Index 0 is background.
    max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

    new_mask = np.zeros_like(mask)
    new_mask[labels == max_label] = 1
    return new_mask


def predict_and_submit():
    print("Starting inference...")
    device = Config.DEVICE

    # Load Model
    model = HRNetSegmentation(num_classes=Config.NUM_CLASSES, pretrained=False).to(
        device
    )
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # Load Test Metadata
    df_test = pd.read_csv(Config.TEST_METADATA_PATH)
    test_dataset = MedicalDataset(df_test, phase="test", transform=None)

    # We iterate manually to handle post-processing per case if needed,
    # but for simplicity and speed we process slice by slice and apply 2D LCC.
    # Batch size 1 for variable sizes.
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=Config.NUM_WORKERS
    )

    results = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            images = batch["image"].to(device)
            orig_shape = batch["orig_shape"].numpy()[0]  # (h, w)
            img_id = batch["id"][0]

            # Pad for HRNet
            h, w = images.shape[2], images.shape[3]
            pad_h = (32 - h % 32) % 32
            pad_w = (32 - w % 32) % 32

            if pad_h > 0 or pad_w > 0:
                images = F.pad(images, (0, pad_w, 0, pad_h), mode="constant", value=0)

            # Predict
            with autocast():
                outputs = model(images)

            # Crop padding
            outputs = outputs[:, :, :h, :w]
            probs = torch.sigmoid(outputs).cpu().numpy()[0]  # (C, H, W)

            # Resize back to original resolution
            # probs shape is (C, Resampled_H, Resampled_W)
            # We need (C, Orig_H, Orig_W)

            final_masks = []
            for c in range(Config.NUM_CLASSES):
                prob_map = probs[c]

                # Resize to original
                if prob_map.shape != tuple(orig_shape):
                    prob_map = cv2.resize(
                        prob_map,
                        (orig_shape[1], orig_shape[0]),
                        interpolation=cv2.INTER_LINEAR,
                    )

                mask = (prob_map > 0.5).astype(np.uint8)

                # Post-processing: Largest Connected Component (2D)
                mask = keep_largest_connected_component(mask)

                rle = rle_encode(mask)
                final_masks.append(rle)

            # Append to results
            # Classes: large_bowel, small_bowel, stomach
            results.append(
                {"id": img_id, "class": "large_bowel", "predicted": final_masks[0]}
            )
            results.append(
                {"id": img_id, "class": "small_bowel", "predicted": final_masks[1]}
            )
            results.append(
                {"id": img_id, "class": "stomach", "predicted": final_masks[2]}
            )

    # Save submission
    submission_df = pd.DataFrame(results)
    # Ensure columns order
    submission_df = submission_df[["id", "class", "predicted"]]
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    # Ensure reproducibility
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Run pipeline
    train_model()
    predict_and_submit()

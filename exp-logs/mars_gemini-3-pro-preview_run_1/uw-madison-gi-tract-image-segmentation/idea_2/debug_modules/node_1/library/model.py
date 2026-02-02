import os
import glob
import re
import gc
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided library files
from library.config import Config
from library.utils import (
    seed_everything,
    rle_encode,
    rle_decode,
    compute_metrics,
    keep_largest_component_3d,
)
from library.losses import BCEDiceLoss


class DecoderBlock(nn.Module):
    """
    Standard U-Net decoder block: UpSample -> Concat -> ConvBlock.
    """

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            # Handle potential padding issues if dimensions don't match exactly
            if x.shape != skip.shape:
                x = F.interpolate(
                    x, size=skip.shape[2:], mode="bilinear", align_corners=True
                )
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNet25D(nn.Module):
    """
    2.5D U-Net with EfficientNet-B1 backbone.
    Takes 3 slices (z, z-1, z-2) as input channels.
    """

    def __init__(
        self, backbone_name=Config.BACKBONE, classes=Config.NUM_CLASSES, pretrained=True
    ):
        super().__init__()

        # Encoder: EfficientNet-B1
        # features_only=True returns intermediate features
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            in_chans=Config.IN_CHANNELS,
        )

        # Get channel counts from the encoder
        # Typically for B1: [24, 40, 80, 112, 320] (indices 0-4)
        enc_channels = self.encoder.feature_info.channels()

        # Decoder
        # We build the decoder from bottom (deepest) to top
        # Center: enc_channels[4] -> Decoder 4

        # Block 4: Up 320 -> 112 (Concat with enc[3])
        self.decoder4 = DecoderBlock(enc_channels[4], enc_channels[3], 256)
        # Block 3: Up 256 -> 80 (Concat with enc[2])
        self.decoder3 = DecoderBlock(256, enc_channels[2], 128)
        # Block 2: Up 128 -> 40 (Concat with enc[1])
        self.decoder2 = DecoderBlock(128, enc_channels[1], 64)
        # Block 1: Up 64 -> 24 (Concat with enc[0])
        self.decoder1 = DecoderBlock(64, enc_channels[0], 32)

        # Final Block: Up 32 -> Output Size (No skip from stride 1 in efficientnet usually, or raw input)
        # We just upsample and conv to final feature map
        self.final_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder pass
        # features is a list of tensors [c1, c2, c3, c4, c5]
        features = self.encoder(x)

        # Decoder pass
        # features indices: 0 (x2), 1 (x4), 2 (x8), 3 (x16), 4 (x32)

        x = self.decoder4(features[4], features[3])
        x = self.decoder3(x, features[2])
        x = self.decoder2(x, features[1])
        x = self.decoder1(x, features[0])

        logits = self.final_conv(x)

        # Ensure output matches input size exactly
        # (Handling cases where input size isn't perfectly divisible by 32)
        # But Config.IMG_SIZE is 320x320 which is divisible by 32.

        return logits


class Mri25DDataset(Dataset):
    """
    Dataset class for 2.5D MRI Segmentation.
    Loads 3 adjacent slices (z, z-1, z-2) to form a 3-channel image.
    Applies percentile normalization and augmentations.
    """

    def __init__(self, df, transforms=None, mode="train"):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.mode = mode

        # Create a lookup for file paths: (case, day, slice_int) -> file_path
        # This allows O(1) retrieval of adjacent slices
        self.df["slice_int"] = self.df["slice"].astype(int)
        self.lookup = self.df.set_index(["case", "day", "slice_int"])[
            "file_path"
        ].to_dict()

        # Pre-calculate available keys to avoid try-except in loop
        self.available_keys = set(self.lookup.keys())

    def __len__(self):
        return len(self.df)

    def load_slice(self, case, day, slice_idx):
        """Loads a single slice. If not found, returns None."""
        key = (case, day, slice_idx)
        if key in self.available_keys:
            path = os.path.join(Config.INPUT_DIR, self.lookup[key])
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            # Handle 16-bit images or other depths
            if img.dtype == np.uint16:
                img = img.astype(np.float32)
            else:
                img = img.astype(np.float32)

            return img
        return None

    def normalize(self, img):
        """Applies percentile normalization."""
        min_val = np.percentile(img, Config.PERCENTILE_MIN)
        max_val = np.percentile(img, Config.PERCENTILE_MAX)

        if max_val > min_val:
            img = (img - min_val) / (max_val - min_val)
        else:
            img = np.zeros_like(img)

        img = np.clip(img, 0, 1)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        case = row["case"]
        day = row["day"]
        current_slice = row["slice_int"]

        # Load 3 slices: z, z-1, z-2
        imgs = []
        for offset in [0, -1, -2]:
            s_idx = current_slice + offset
            img = self.load_slice(case, day, s_idx)

            # If adjacent slice doesn't exist (boundary), replicate the current/closest one
            if img is None:
                # Try loading the current slice again as fallback
                if len(imgs) > 0:
                    img = imgs[-1]  # Copy previous loaded
                else:
                    # Should not happen if dataset is valid, but fallback to z
                    img = self.load_slice(case, day, current_slice)

            # Normalize immediately
            img = self.normalize(img)
            imgs.append(img)

        # Stack to (H, W, 3)
        # imgs[0] is z, imgs[1] is z-1, imgs[2] is z-2
        img_stack = np.stack(imgs, axis=-1)

        # Resize to Config.IMG_SIZE
        img_stack = cv2.resize(
            img_stack,
            (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
            interpolation=cv2.INTER_LINEAR,
        )

        mask_stack = None
        if self.mode in ["train", "val"]:
            # Load masks
            mask_stack = np.zeros(
                (Config.IMG_SIZE[0], Config.IMG_SIZE[1], 3), dtype=np.float32
            )

            for i, cls_name in enumerate(Config.CLASSES):
                rle = row[cls_name]
                if isinstance(rle, str) and rle != "":
                    # Decode to original size
                    mask = rle_decode(rle, (row["height"], row["width"]))
                    # Resize to model input size
                    mask = cv2.resize(
                        mask,
                        (Config.IMG_SIZE[1], Config.IMG_SIZE[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    mask_stack[:, :, i] = mask

        # Augmentations
        if self.transforms:
            if mask_stack is not None:
                augmented = self.transforms(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]
            else:
                augmented = self.transforms(image=img_stack)
                img_stack = augmented["image"]
        else:
            # Basic ToTensor if no transforms provided
            t = ToTensorV2()
            if mask_stack is not None:
                augmented = t(image=img_stack, mask=mask_stack)
                img_stack = augmented["image"]
                mask_stack = augmented["mask"]
            else:
                augmented = t(image=img_stack)
                img_stack = augmented["image"]

        if self.mode in ["train", "val"]:
            # Mask shape: (3, H, W), Image shape: (3, H, W)
            return img_stack, mask_stack, row["id"]
        else:
            return img_stack, row["id"]


def get_transforms(data="train"):
    if data == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5
                ),
                A.OneOf(
                    [
                        A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                        A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                    ],
                    p=0.25,
                ),
                ToTensorV2(),
            ]
        )
    elif data == "val" or data == "test":
        return A.Compose(
            [
                ToTensorV2(),
            ]
        )


def load_data(load_cached_data=True):
    """
    Loads metadata. Caching is implemented via parquet files in the metadata directory
    which are already pre-generated by the task description.
    We just read them.
    """
    # The prompt says metadata is already in ./metadata/
    # We will just read them.
    # If we needed to process raw files, we would implement caching here.

    train_df = pd.read_csv(Config.TRAIN_CSV, keep_default_na=False)
    val_df = pd.read_csv(Config.VAL_CSV, keep_default_na=False)
    test_df = pd.read_csv(Config.TEST_CSV, keep_default_na=False)

    return train_df, val_df, test_df


def train_model(debug=False):
    """
    Main training loop.
    """
    seed_everything(Config.SEED)

    # Load Data
    train_df, val_df, _ = load_data()

    if debug:
        train_df = train_df.head(200)
        val_df = val_df.head(100)
        epochs = 2
    else:
        epochs = Config.EPOCHS

    # Datasets & Loaders
    train_dataset = Mri25DDataset(
        train_df, transforms=get_transforms("train"), mode="train"
    )
    val_dataset = Mri25DDataset(val_df, transforms=get_transforms("val"), mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model, Loss, Optimizer
    model = UNet25D().to(Config.DEVICE)
    optimizer = AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)
    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    best_score = 0.0
    best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Training Loop
        # Using tqdm is allowed but prompt says "Only print the required information... do not print progress bars"
        # I will suppress tqdm or just print summary at end of epoch

        for images, masks, _ in train_loader:
            images = images.to(Config.DEVICE)
            masks = masks.to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        # Validation Loop (3D Reconstruction Strategy)
        model.eval()

        # To compute 3D metrics, we need to aggregate predictions by case
        # We'll store predictions and ground truths in a dictionary: case_id -> list of slices
        val_preds_dict = {}
        val_gt_dict = {}

        # We need to know original dimensions to resize back for metric calculation
        # But for speed in validation, we can compute metrics at model resolution (320x320)
        # or resize. The prompt implies Hausdorff should be on volumes.
        # Let's collect data first.

        with torch.no_grad():
            for images, masks, ids in val_loader:
                images = images.to(Config.DEVICE)
                outputs = model(images)
                outputs = torch.sigmoid(outputs)

                # Move to CPU
                preds = (outputs > 0.5).float().cpu().numpy()
                gts = masks.cpu().numpy()

                for i, slice_id in enumerate(ids):
                    # slice_id format: caseXXX_dayYY_slice_ZZZZ
                    case_day = "_".join(slice_id.split("_")[:2])
                    slice_num = int(slice_id.split("_")[-1])

                    if case_day not in val_preds_dict:
                        val_preds_dict[case_day] = []
                        val_gt_dict[case_day] = []

                    val_preds_dict[case_day].append((slice_num, preds[i]))
                    val_gt_dict[case_day].append((slice_num, gts[i]))

        # Compute Metrics per Case
        case_scores = []

        for case_day in val_preds_dict:
            # Sort by slice number
            preds_list = sorted(val_preds_dict[case_day], key=lambda x: x[0])
            gt_list = sorted(val_gt_dict[case_day], key=lambda x: x[0])

            # Stack to (D, C, H, W) -> (C, D, H, W)
            vol_pred = np.stack([x[1] for x in preds_list], axis=1)
            vol_gt = np.stack([x[1] for x in gt_list], axis=1)

            # Compute metric for each class
            # Classes: 0: Large Bowel, 1: Small Bowel, 2: Stomach
            current_case_score = 0

            for c in range(Config.NUM_CLASSES):
                p = vol_pred[c]  # (D, H, W)
                g = vol_gt[c]  # (D, H, W)

                # Post-processing: Keep largest component (optional during val, but good for consistency)
                # Doing it in validation to match inference strategy
                # Note: This might be slow. If too slow, disable for val.
                # Given 24h budget, we can afford it.
                # p = keep_largest_component_3d(p)
                # (Skipping CCA in val loop for speed, relying on raw prediction quality for model selection)

                metrics = compute_metrics(p, g)
                current_case_score += metrics["score"]

            case_scores.append(current_case_score / Config.NUM_CLASSES)

        avg_val_score = np.mean(case_scores)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Score: {avg_val_score:.4f}"
        )

        if avg_val_score > best_score:
            best_score = avg_val_score
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best model saved! Score: {best_score:.4f}")

    print(f"Training complete. Best Validation Score: {best_score:.4f}")


def predict():
    """
    Inference pipeline.
    Generates predictions for the test set, applies 3D post-processing,
    and creates the submission file.
    """
    seed_everything(Config.SEED)

    # Load Data
    _, _, test_df = load_data()

    # Dataset & Loader
    test_dataset = Mri25DDataset(
        test_df, transforms=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE * 2,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = UNet25D().to(Config.DEVICE)
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print("No checkpoint found! Please train the model first.")
        return

    model.load_state_dict(torch.load(checkpoint_path, map_location=Config.DEVICE))
    model.eval()

    print("Starting inference...")

    # Store predictions: case_day -> list of (slice_num, slice_id, original_h, original_w, pred_mask)
    results_dict = {}

    with torch.no_grad():
        for images, ids in test_loader:
            images = images.to(Config.DEVICE)
            outputs = model(images)
            outputs = torch.sigmoid(outputs)

            preds = (outputs > 0.5).float().cpu().numpy()

            for i, slice_id in enumerate(ids):
                # Retrieve metadata for resizing
                # We need to query the original dataframe for H/W
                # This is a bit slow, so we use the row from the dataset if possible
                # But dataset __getitem__ returns processed tensors.
                # We'll parse ID.

                # slice_id: caseXXX_dayYY_slice_ZZZZ
                case_day = "_".join(slice_id.split("_")[:2])
                slice_num = int(slice_id.split("_")[-1])

                if case_day not in results_dict:
                    results_dict[case_day] = []

                results_dict[case_day].append(
                    {
                        "slice_num": slice_num,
                        "id": slice_id,
                        "mask": preds[i],  # (3, H, W)
                    }
                )

    # Process per case (3D CCA and RLE)
    submission_rows = []

    print("Post-processing and encoding...")

    # Retrieve original dimensions from test_df for resizing
    # Create lookup (id -> (w, h))
    dims_lookup = test_df.set_index("id")[["width", "height"]].to_dict("index")

    for case_day, slices in results_dict.items():
        # Sort by slice number to form volume
        slices.sort(key=lambda x: x["slice_num"])

        # Stack volumes: (D, 3, H, W) -> (3, D, H, W)
        vol = np.stack([s["mask"] for s in slices], axis=1)

        # Apply 3D CCA per class
        for c in range(Config.NUM_CLASSES):
            # Extract class volume (D, H, W)
            class_vol = vol[c]

            # Keep largest component
            class_vol = keep_largest_component_3d(class_vol)

            # Update volume
            vol[c] = class_vol

        # Iterate back through slices to encode
        for i, s_info in enumerate(slices):
            slice_id = s_info["id"]
            orig_w = dims_lookup[slice_id]["width"]
            orig_h = dims_lookup[slice_id]["height"]

            for c_idx, class_name in enumerate(Config.CLASSES):
                # Get mask for this slice/class (H_model, W_model)
                mask_slice = vol[c_idx, i, :, :]

                # Resize back to original resolution
                # Use nearest neighbor to keep binary
                mask_resized = cv2.resize(
                    mask_slice, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
                )

                # Encode
                rle = rle_encode(mask_resized)

                submission_rows.append(
                    {"id": slice_id, "class": class_name, "predicted": rle}
                )

    # Save Submission
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(Config.SUBMISSION_FILE, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE}")

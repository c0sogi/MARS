import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import from provided libraries
from library.utils import set_seed, calculate_map_score
from library.losses import BCEDiceLoss, LovaszLoss

# -----------------------------------------------------------------------------
# 1. Dataset
# -----------------------------------------------------------------------------


class SaltDataset(Dataset):
    def __init__(self, metadata_csv, transform=None, mode="train", input_dir="./input"):
        """
        Args:
            metadata_csv (str): Path to the metadata CSV file.
            transform (albumentations.Compose): Augmentations.
            mode (str): 'train', 'val', or 'test'.
            input_dir (str): Root directory for input data.
        """
        self.df = pd.read_csv(metadata_csv)
        self.transform = transform
        self.mode = mode
        self.input_dir = input_dir

        # Depth normalization constants (min/max from dataset analysis)
        self.z_min = 50.0
        self.z_max = 960.0

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load Image
        img_path = os.path.join(self.input_dir, row["image_path"])
        # Load as grayscale
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")

        # Load Mask (if available)
        mask = None
        if self.mode in ["train", "val"] and pd.notna(row.get("mask_path")):
            mask_path = os.path.join(self.input_dir, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # Threshold to binary 0-1
            mask = (mask > 127).astype(np.float32)

        # Depth Processing
        z = row["z"]
        z_norm = (z - self.z_min) / (self.z_max - self.z_min)

        # Padding to 128x128 (Reflection Padding)
        # Original is 101x101
        orig_h, orig_w = image.shape
        target_h, target_w = 128, 128
        pad_h = target_h - orig_h
        pad_w = target_w - orig_w
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        image = cv2.copyMakeBorder(
            image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
        )
        if mask is not None:
            mask = cv2.copyMakeBorder(
                mask, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT
            )

        # Augmentations
        if self.transform:
            if mask is not None:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                augmented = self.transform(image=image)
                image = augmented["image"]

        # Ensure tensors
        if not isinstance(image, torch.Tensor):
            image = image.astype(np.float32) / 255.0
            image = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)
        else:
            # Albumentations ToTensorV2 usually keeps dtype, ensure float 0-1
            if image.dtype == torch.uint8:
                image = image.float() / 255.0

        if mask is not None and not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).unsqueeze(0)  # (1, H, W)
        elif mask is not None:
            mask = mask.float()

        # Depth Fusion: Create dense depth channel
        _, h, w = image.shape
        depth_channel = torch.full((1, h, w), z_norm, dtype=torch.float32)

        # Concatenate: Input becomes (2, 128, 128)
        image = torch.cat([image, depth_channel], dim=0)

        result = {"image": image, "id": row["id"]}
        if mask is not None:
            result["mask"] = mask

        return result


# -----------------------------------------------------------------------------
# 2. Model Architecture
# -----------------------------------------------------------------------------


class SCSEBlock(nn.Module):
    """
    Concurrent Spatial and Channel Squeeze & Excitation Block.
    """

    def __init__(self, in_channels, reduction=16):
        super().__init__()
        mid_channels = max(1, in_channels // reduction)

        # Channel SE
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, mid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial SE
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    """
    Standard Double Convolution Block: Conv-BN-ReLU-Conv-BN-ReLU
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class HyperColumnUNet(nn.Module):
    """
    U-Net with SCSE blocks and a Hypercolumn head.
    Expects 2 input channels (Image + Depth).
    """

    def __init__(self, input_channels=2, num_classes=1, base_filters=32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(input_channels, base_filters)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = ConvBlock(base_filters, base_filters * 2)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = ConvBlock(base_filters * 2, base_filters * 4)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.enc4 = ConvBlock(base_filters * 4, base_filters * 8)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Center
        self.center = ConvBlock(base_filters * 8, base_filters * 16)

        # Decoder with SCSE
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec4 = ConvBlock(base_filters * 16, base_filters * 8)
        self.scse4 = SCSEBlock(base_filters * 8)

        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec3 = ConvBlock(base_filters * 8, base_filters * 4)
        self.scse3 = SCSEBlock(base_filters * 4)

        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 2, stride=2)
        self.dec2 = ConvBlock(base_filters * 4, base_filters * 2)
        self.scse2 = SCSEBlock(base_filters * 2)

        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, 2, stride=2)
        self.dec1 = ConvBlock(base_filters * 2, base_filters)
        self.scse1 = SCSEBlock(base_filters)

        # Hypercolumn Head
        # Concatenates upsampled outputs from all decoder levels
        # Channels: (256 + 128 + 64 + 32) = 480 (if base=32)
        total_channels = (8 + 4 + 2 + 1) * base_filters
        self.final_conv = nn.Conv2d(total_channels, num_classes, 1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)

        e2 = self.enc2(p1)
        p2 = self.pool2(e2)

        e3 = self.enc3(p2)
        p3 = self.pool3(e3)

        e4 = self.enc4(p3)
        p4 = self.pool4(e4)

        # Center
        c = self.center(p4)

        # Decoder
        d4 = self.up4(c)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.scse4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.scse3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.scse2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        d1 = self.scse1(d1)

        # Hypercolumns
        target_size = x.size()[2:]

        hc4 = F.interpolate(d4, size=target_size, mode="bilinear", align_corners=True)
        hc3 = F.interpolate(d3, size=target_size, mode="bilinear", align_corners=True)
        hc2 = F.interpolate(d2, size=target_size, mode="bilinear", align_corners=True)
        hc1 = d1  # Already at target size

        hypercol = torch.cat([hc4, hc3, hc2, hc1], dim=1)

        out = self.final_conv(hypercol)

        return out


# -----------------------------------------------------------------------------
# 3. Training Function
# -----------------------------------------------------------------------------


def train_model(
    train_metadata_path="./metadata/train.csv",
    val_metadata_path="./metadata/val.csv",
    batch_size=32,
    epochs=50,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
    save_path="./working/idea_3/best_model.pth",
):
    """
    Trains the HyperColumnUNet model using a two-stage loss strategy.
    Stage 1: BCE + Dice Loss (Convergence)
    Stage 2: Lovasz-Softmax Loss (Fine-tuning)
    """
    set_seed(42)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Transforms
    train_transform = A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5
            ),
            ToTensorV2(),
        ]
    )
    val_transform = A.Compose([ToTensorV2()])

    # Datasets
    train_dataset = SaltDataset(
        train_metadata_path, transform=train_transform, mode="train"
    )
    val_dataset = SaltDataset(val_metadata_path, transform=val_transform, mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model
    model = HyperColumnUNet(input_channels=2, num_classes=1, base_filters=32)
    model = model.to(device)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Losses
    criterion_stage1 = BCEDiceLoss()
    criterion_stage2 = LovaszLoss()

    best_map = 0.0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        # Switch loss after 50% of epochs
        use_lovasz = epoch >= (epochs // 2)
        current_criterion = criterion_stage2 if use_lovasz else criterion_stage1
        loss_name = "Lovasz" if use_lovasz else "BCE+Dice"

        for batch in train_loader:
            images = batch["image"].to(device)
            masks = batch["mask"].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = current_criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_truths = []

        # Crop parameters to revert 128x128 -> 101x101
        pad_top, pad_left = 13, 13
        orig_h, orig_w = 101, 101

        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                masks = batch["mask"].to(device)

                outputs = model(images)
                loss = current_criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)

                probs = torch.sigmoid(outputs).cpu().numpy()
                true_masks = masks.cpu().numpy()

                for i in range(probs.shape[0]):
                    # Crop center
                    pred_crop = probs[
                        i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                    ]
                    true_crop = true_masks[
                        i, 0, pad_top : pad_top + orig_h, pad_left : pad_left + orig_w
                    ]

                    all_preds.append(pred_crop)
                    all_truths.append(true_crop)

        val_loss /= len(val_dataset)

        # Calculate mAP (Threshold at 0.5 for IoU calculation)
        bin_preds = [p > 0.5 for p in all_preds]
        bin_truths = [t > 0.5 for t in all_truths]
        val_map = calculate_map_score(bin_preds, bin_truths)

        print(
            f"Epoch {epoch+1}/{epochs} | Loss ({loss_name}): Train={train_loss:.4f}, Val={val_loss:.4f} | mAP: {val_map:.10f}"
        )

        scheduler.step(val_map)

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"  New best model saved! mAP: {val_map:.10f}")

    print(f"Training complete. Best mAP: {best_map:.10f}")
    return model

import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, rle_encode, metric_iou


# ==========================================
# Dataset
# ==========================================
class SaltDataset(Dataset):
    def __init__(self, metadata_path, config, mode="train"):
        """
        Args:
            metadata_path (str): Path to the csv file (train.csv, val.csv, test.csv).
            config (Config): Configuration object.
            mode (str): 'train', 'val', or 'test'.
        """
        self.df = pd.read_csv(metadata_path)
        self.config = config
        self.mode = mode
        self.input_root = config.INPUT_ROOT

        # Pre-calculate padding
        h, w = config.ORIG_SHAPE
        target_h, target_w = config.INPUT_SHAPE
        pad_h = target_h - h
        pad_w = target_w - w
        self.pad_top = pad_h // 2
        self.pad_bottom = pad_h - self.pad_top
        self.pad_left = pad_w // 2
        self.pad_right = pad_w - self.pad_left

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.input_root, row["image_path"])

        # Load Image (Grayscale)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback for missing images (should not happen based on EDA)
            image = np.zeros(self.config.ORIG_SHAPE, dtype=np.uint8)

        # Pad Image
        image = cv2.copyMakeBorder(
            image,
            self.pad_top,
            self.pad_bottom,
            self.pad_left,
            self.pad_right,
            cv2.BORDER_REFLECT,
        )

        # Normalize Image [0, 1]
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=0)  # (1, H, W)

        # Process Depth
        z = row["z"]
        z_norm = (z - self.config.DEPTH_MEAN) / self.config.DEPTH_STD
        z_tensor = torch.tensor([z_norm], dtype=torch.float32)

        # Load Mask (if available)
        mask_tensor = torch.zeros((1, *self.config.INPUT_SHAPE), dtype=torch.float32)
        if self.mode in ["train", "val"]:
            mask_path = os.path.join(self.input_root, row["mask_path"])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                mask = cv2.copyMakeBorder(
                    mask,
                    self.pad_top,
                    self.pad_bottom,
                    self.pad_left,
                    self.pad_right,
                    cv2.BORDER_REFLECT,
                )
                mask = (mask > 127).astype(np.float32)
                mask = np.expand_dims(mask, axis=0)
                mask_tensor = torch.from_numpy(mask)

        # Augmentation (Train only)
        image_tensor = torch.from_numpy(image)
        if self.mode == "train":
            # Horizontal Flip
            if np.random.rand() > 0.5:
                image_tensor = torch.flip(image_tensor, [2])
                mask_tensor = torch.flip(mask_tensor, [2])

        if self.mode == "test":
            return image_tensor, z_tensor, row["id"]
        else:
            return image_tensor, z_tensor, mask_tensor


# ==========================================
# Model Components
# ==========================================
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # LinkNet Decoder style: 1x1 (reduce) -> Deconv (upsample) -> 1x1 (expand/restore)
        # We adapt slightly: 1x1 -> Deconv -> 1x1 to match output channels
        mid_channels = in_channels // 4

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.deconv = nn.ConvTranspose2d(
            mid_channels,
            mid_channels,
            3,
            stride=2,
            padding=1,
            output_padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.deconv(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = self.relu(x)
        return x


class DepthLinkNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=1):
        super().__init__()

        # Initial Block
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Encoder (ResNet-like)
        self.enc1 = EncoderBlock(32, 64, stride=2)  # 128 -> 64
        self.enc2 = EncoderBlock(64, 128, stride=2)  # 64 -> 32
        self.enc3 = EncoderBlock(128, 256, stride=2)  # 32 -> 16
        self.enc4 = EncoderBlock(256, 512, stride=2)  # 16 -> 8

        # Depth Injection
        self.depth_channels = 32
        self.depth_fc = nn.Sequential(
            nn.Linear(1, self.depth_channels), nn.ReLU(inplace=True)
        )

        # Decoder
        # Input to Dec4 is Enc4_out + Depth
        self.dec4 = DecoderBlock(512 + self.depth_channels, 256)
        self.dec3 = DecoderBlock(256, 128)
        self.dec2 = DecoderBlock(128, 64)
        self.dec1 = DecoderBlock(64, 32)

        # Final Head
        self.final_conv = nn.Conv2d(32, num_classes, 1)

    def forward(self, x, z):
        # Encoder
        x0 = self.init_conv(x)  # 32, 128, 128
        e1 = self.enc1(x0)  # 64, 64, 64
        e2 = self.enc2(e1)  # 128, 32, 32
        e3 = self.enc3(e2)  # 256, 16, 16
        e4 = self.enc4(e3)  # 512, 8, 8

        # Depth Injection
        # z: (B, 1) -> (B, depth_ch)
        d = self.depth_fc(z)
        # Expand to (B, depth_ch, 8, 8)
        d = d.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, e4.size(2), e4.size(3))

        bottleneck = torch.cat([e4, d], dim=1)  # 544, 8, 8

        # Decoder with LinkNet Skip Connections (Addition)
        d4 = self.dec4(bottleneck)  # -> 256, 16, 16
        d4 = d4 + e3

        d3 = self.dec3(d4)  # -> 128, 32, 32
        d3 = d3 + e2

        d2 = self.dec2(d3)  # -> 64, 64, 64
        d2 = d2 + e1

        d1 = self.dec1(d2)  # -> 32, 128, 128
        d1 = d1 + x0

        out = self.final_conv(d1)
        return out


# ==========================================
# Manager / Trainer
# ==========================================
class SaltModelManager:
    def __init__(self, config=None):
        self.config = config if config else Config()
        set_seed(self.config.SEED)

        self.device = torch.device(self.config.DEVICE)
        self.model = DepthLinkNet(
            in_channels=self.config.CHANNELS, num_classes=self.config.NUM_CLASSES
        ).to(self.device)

        print(f"Model initialized on {self.device}")

    def criterion(self, pred, target):
        # BCEWithLogits + Dice Loss
        bce = nn.BCEWithLogitsLoss()(pred, target)

        pred_sig = torch.sigmoid(pred)
        smooth = 1.0
        intersection = (pred_sig * target).sum()
        dice = 1 - (
            (2.0 * intersection + smooth) / (pred_sig.sum() + target.sum() + smooth)
        )

        return 0.5 * bce + 0.5 * dice

    def train(self):
        # Data Loaders
        train_ds = SaltDataset(self.config.TRAIN_CSV, self.config, mode="train")
        val_ds = SaltDataset(self.config.VAL_CSV, self.config, mode="val")

        train_loader = DataLoader(
            train_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.EPOCHS, eta_min=1e-6
        )

        best_iou = 0.0
        patience_counter = 0

        print(f"Starting training for {self.config.EPOCHS} epochs...")

        for epoch in range(self.config.EPOCHS):
            self.model.train()
            train_loss = 0.0

            for images, depths, masks in train_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)
                masks = masks.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(images, depths)
                loss = self.criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * images.size(0)

            train_loss /= len(train_ds)

            # Validation
            val_loss, val_iou = self.validate(val_loader)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val IoU: {val_iou:.6f}"
            )

            # Checkpoint
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(self.model.state_dict(), self.config.CHECKPOINT_PATH)
                print(f"  -> New Best Model Saved! IoU: {best_iou:.6f}")
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val IoU: {best_iou:.6f}")

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0
        total_iou = 0.0

        with torch.no_grad():
            for images, depths, masks in loader:
                images = images.to(self.device)
                depths = depths.to(self.device)
                masks = masks.to(self.device)

                outputs = self.model(images, depths)
                loss = self.criterion(outputs, masks)
                total_loss += loss.item() * images.size(0)

                # Calculate metric
                probs = torch.sigmoid(outputs)
                batch_iou = metric_iou(masks, probs, threshold=self.config.THRESHOLD)
                total_iou += batch_iou * images.size(0)

        return total_loss / len(loader.dataset), total_iou / len(loader.dataset)

    def generate_submission(self):
        print("Generating submission...")

        # Load best model
        if os.path.exists(self.config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(self.config.CHECKPOINT_PATH, map_location=self.device)
            )
            print("Loaded best checkpoint.")
        else:
            print("Warning: No checkpoint found, using current model weights.")

        self.model.eval()

        test_ds = SaltDataset(self.config.TEST_CSV, self.config, mode="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=self.config.BATCH_SIZE * 2,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
        )

        results = []

        # Calculate cropping indices to revert padding
        # 128x128 -> 101x101
        h, w = self.config.ORIG_SHAPE
        target_h, target_w = self.config.INPUT_SHAPE
        pad_top = (target_h - h) // 2
        pad_left = (target_w - w) // 2

        with torch.no_grad():
            for images, depths, ids in test_loader:
                images = images.to(self.device)
                depths = depths.to(self.device)

                outputs = self.model(images, depths)
                probs = torch.sigmoid(outputs)

                # Move to CPU
                probs = probs.cpu().numpy()

                for i, img_id in enumerate(ids):
                    # Crop back to 101x101
                    prob_map = probs[i, 0, :, :]
                    mask_map = prob_map[pad_top : pad_top + h, pad_left : pad_left + w]

                    # Threshold
                    binary_mask = (mask_map > self.config.THRESHOLD).astype(np.uint8)

                    # RLE Encode
                    rle = rle_encode(binary_mask)
                    results.append({"id": img_id, "rle_mask": rle})

        # Create DataFrame
        sub_df = pd.DataFrame(results)

        # Ensure directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        # Save
        sub_df.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

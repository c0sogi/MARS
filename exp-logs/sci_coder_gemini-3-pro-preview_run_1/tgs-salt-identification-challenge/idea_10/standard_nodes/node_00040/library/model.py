import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import provided library functions
from library.utils import set_seed, rle_encode, rle_decode, do_kaggle_metric
from library.loss import CompoundLoss

# -------------------------------------------------------------------------
# 1. Model Architecture
# -------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
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


class SCSEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SCSEBlock, self).__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class DeepResUNet(nn.Module):
    def __init__(self, in_channels=2, classes=1):
        super(DeepResUNet, self).__init__()

        # Encoder
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )  # 128x128

        self.enc1 = ResBlock(64, 128, stride=2)  # 64x64
        self.enc2 = ResBlock(128, 256, stride=2)  # 32x32
        self.enc3 = ResBlock(256, 512, stride=2)  # 16x16
        self.enc4 = ResBlock(512, 1024, stride=2)  # 8x8

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up4 = nn.Conv2d(1024, 512, 3, padding=1)
        self.dec4 = ResBlock(1024, 512)
        self.scse4 = SCSEBlock(512)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up3 = nn.Conv2d(512, 256, 3, padding=1)
        self.dec3 = ResBlock(512, 256)
        self.scse3 = SCSEBlock(256)
        self.aux_head2 = nn.Conv2d(256, classes, 1)  # 32x32

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up2 = nn.Conv2d(256, 128, 3, padding=1)
        self.dec2 = ResBlock(256, 128)
        self.scse2 = SCSEBlock(128)
        self.aux_head1 = nn.Conv2d(128, classes, 1)  # 64x64

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv_up1 = nn.Conv2d(128, 64, 3, padding=1)
        self.dec1 = ResBlock(128, 64)
        self.scse1 = SCSEBlock(64)

        self.final_conv = nn.Conv2d(64, classes, 1)  # 128x128

    def forward(self, x):
        # Encoder
        x0 = self.init_conv(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)

        # Decoder
        d4 = self.up4(x4)
        d4 = self.conv_up4(d4)
        d4 = torch.cat([d4, x3], dim=1)
        d4 = self.dec4(d4)
        d4 = self.scse4(d4)

        d3 = self.up3(d4)
        d3 = self.conv_up3(d3)
        d3 = torch.cat([d3, x2], dim=1)
        d3 = self.dec3(d3)
        d3 = self.scse3(d3)
        aux2 = self.aux_head2(d3)

        d2 = self.up2(d3)
        d2 = self.conv_up2(d2)
        d2 = torch.cat([d2, x1], dim=1)
        d2 = self.dec2(d2)
        d2 = self.scse2(d2)
        aux1 = self.aux_head1(d2)

        d1 = self.up1(d2)
        d1 = self.conv_up1(d1)
        d1 = torch.cat([d1, x0], dim=1)
        d1 = self.dec1(d1)
        d1 = self.scse1(d1)

        logits = self.final_conv(d1)

        return {"logits": logits, "aux_32": aux2, "aux_64": aux1}


# -------------------------------------------------------------------------
# 2. Data Processing and Caching
# -------------------------------------------------------------------------


def preprocess_and_cache(metadata_path, cache_dir, load_cached_data=True, mode="train"):
    """
    Loads images, pre-processes them (pad, depth fusion), and caches as .npy.
    """
    os.makedirs(cache_dir, exist_ok=True)

    ids_path = os.path.join(cache_dir, f"{mode}_ids.npy")
    images_path = os.path.join(cache_dir, f"{mode}_images.npy")
    masks_path = os.path.join(cache_dir, f"{mode}_masks.npy")

    if load_cached_data and os.path.exists(images_path) and os.path.exists(ids_path):
        if mode == "test" or os.path.exists(masks_path):
            print(f"Loading cached {mode} data from {cache_dir}...")
            ids = np.load(ids_path, allow_pickle=True)
            images = np.load(images_path)
            if mode != "test":
                masks = np.load(masks_path)
                return ids, images, masks
            return ids, images, None

    print(f"Processing {mode} data from scratch...")
    df = pd.read_csv(metadata_path)

    img_list = []
    mask_list = []
    id_list = []

    # Pre-calculate depth stats for normalization (Global min/max from analysis)
    DEPTH_MIN = 51.0
    DEPTH_MAX = 959.0

    input_dir = "./input"

    for idx, row in df.iterrows():
        id_ = row["id"]
        z = row["z"]

        # Load Image
        img_path = os.path.join(input_dir, row["image_path"])
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Normalize Image
        img = img.astype(np.float32) / 255.0

        # Load Mask if train/val
        mask = None
        if mode != "test":
            mask_path = os.path.join(input_dir, row["mask_path"])
            mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask_img.astype(np.float32) / 255.0
            mask = (mask > 0.5).astype(np.float32)

        # Reflection Padding 101 -> 128
        # Pad: (128-101) = 27. Top=13, Bottom=14. Left=13, Right=14.
        pad_h = 128 - 101
        pad_w = 128 - 101
        p_top = pad_h // 2
        p_bot = pad_h - p_top
        p_left = pad_w // 2
        p_right = pad_w - p_left

        img_padded = cv2.copyMakeBorder(
            img, p_top, p_bot, p_left, p_right, cv2.BORDER_REFLECT_101
        )

        if mask is not None:
            mask_padded = cv2.copyMakeBorder(
                mask, p_top, p_bot, p_left, p_right, cv2.BORDER_REFLECT_101
            )
            mask_list.append(mask_padded)

        # Depth Channel
        z_norm = (z - DEPTH_MIN) / (DEPTH_MAX - DEPTH_MIN)
        depth_channel = np.full_like(img_padded, z_norm)

        # Stack: (128, 128, 2)
        combined = np.stack([img_padded, depth_channel], axis=-1)

        img_list.append(combined)
        id_list.append(id_)

    images = np.array(img_list, dtype=np.float32)
    ids = np.array(id_list)

    np.save(ids_path, ids)
    np.save(images_path, images)

    if mode != "test":
        masks = np.array(mask_list, dtype=np.float32)
        # Expand dims for masks: (N, 128, 128) -> (N, 128, 128, 1)
        masks = masks[..., np.newaxis]
        np.save(masks_path, masks)
        return ids, images, masks

    return ids, images, None


class SaltDataset(Dataset):
    def __init__(self, images, masks=None, transform=None):
        self.images = images
        self.masks = masks
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]  # (128, 128, 2)

        if self.masks is not None:
            mask = self.masks[idx]  # (128, 128, 1)

            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1))
                mask = torch.from_numpy(mask.transpose(2, 0, 1))

            # Ensure mask is (C, H, W)
            if mask.ndim == 3 and mask.shape[-1] == 1:
                mask = mask.permute(2, 0, 1)

            return image, mask
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            else:
                image = torch.from_numpy(image.transpose(2, 0, 1))
            return image, ""


# -------------------------------------------------------------------------
# 3. Training and Inference
# -------------------------------------------------------------------------


def train_model(epochs=150, batch_size=32, num_workers=4, debug=False):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Paths
    cache_dir = "./working/idea_10"
    checkpoint_dir = "./working/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load Data
    train_ids, train_imgs, train_masks = preprocess_and_cache(
        "./metadata/train.csv", cache_dir, mode="train"
    )
    val_ids, val_imgs, val_masks = preprocess_and_cache(
        "./metadata/val.csv", cache_dir, mode="val"
    )

    if debug:
        train_imgs = train_imgs[:100]
        train_masks = train_masks[:100]
        val_imgs = val_imgs[:20]
        val_masks = val_masks[:20]
        epochs = 2

    # Transforms
    train_transform = A.Compose([A.HorizontalFlip(p=0.5), ToTensorV2()])
    val_transform = A.Compose([ToTensorV2()])

    train_dataset = SaltDataset(train_imgs, train_masks, transform=train_transform)
    val_dataset = SaltDataset(val_imgs, val_masks, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # Model
    model = DeepResUNet(in_channels=2).to(device)

    # Optimization
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-5)

    criterion = CompoundLoss()
    criterion_aux = nn.BCEWithLogitsLoss()

    best_map = 0.0

    print("Starting training...")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # Resize masks for aux heads
            mask_32 = F.interpolate(masks, size=(32, 32), mode="nearest")
            mask_64 = F.interpolate(masks, size=(64, 64), mode="nearest")

            loss_main = criterion(outputs, masks)
            loss_aux1 = criterion_aux(outputs["aux_64"], mask_64)
            loss_aux2 = criterion_aux(outputs["aux_32"], mask_32)

            loss = loss_main + 0.5 * loss_aux1 + 0.5 * loss_aux2

            loss.backward()
            optimizer.step()

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)

                # Loss
                loss_main = criterion(outputs, masks)
                loss_aux1 = criterion_aux(
                    outputs["aux_64"],
                    F.interpolate(masks, size=(64, 64), mode="nearest"),
                )
                loss_aux2 = criterion_aux(
                    outputs["aux_32"],
                    F.interpolate(masks, size=(32, 32), mode="nearest"),
                )
                loss = loss_main + 0.5 * loss_aux1 + 0.5 * loss_aux2

                val_loss += loss.item() * images.size(0)

                # Predictions for Metric
                logits = outputs["logits"]
                final_probs = torch.sigmoid(logits)

                # Unpad to 101x101
                # Pad was: Top=13, Bot=14, Left=13, Right=14
                # 128 -> 101
                final_probs = final_probs[:, :, 13:114, 13:114]
                masks_cropped = masks[:, :, 13:114, 13:114]

                all_preds.append(final_probs.cpu().numpy())
                all_targets.append(masks_cropped.cpu().numpy())

        val_loss /= len(val_dataset)
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Calculate mAP
        map_score = do_kaggle_metric(all_preds, all_targets, threshold=0.5)

        print(
            f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mAP: {map_score}"
        )

        # Save Best Model and Cycle Snapshots
        if map_score > best_map:
            best_map = map_score
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_model.pth")
            )

        if (epoch + 1) == 100:
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_cycle_2.pth")
            )
        if (epoch + 1) == 150:
            torch.save(
                model.state_dict(), os.path.join(checkpoint_dir, "best_cycle_3.pth")
            )


def predict_and_submit():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_dir = "./working/idea_10"
    checkpoint_dir = "./working/checkpoints"

    # Load Test Data
    test_ids, test_imgs, _ = preprocess_and_cache(
        "./metadata/test.csv", cache_dir, mode="test"
    )

    test_dataset = SaltDataset(
        test_imgs, masks=None, transform=A.Compose([ToTensorV2()])
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # Load Models for Ensemble
    paths = [
        os.path.join(checkpoint_dir, "best_cycle_2.pth"),
        os.path.join(checkpoint_dir, "best_cycle_3.pth"),
    ]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        paths = [os.path.join(checkpoint_dir, "best_model.pth")]

    models = []
    for p in paths:
        m = DeepResUNet(in_channels=2).to(device)
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        models.append(m)

    print(f"Ensembling {len(models)} models...")

    rle_list = []
    id_list = []

    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            images = images.to(device)

            # TTA: Original and Flip
            images_flip = torch.flip(images, [3])

            batch_preds = []

            for model in models:
                # Forward
                out = model(images)
                out_flip = model(images_flip)

                prob = torch.sigmoid(out["logits"])
                prob_flip = torch.sigmoid(out_flip["logits"])

                # Undo flip
                prob_flip = torch.flip(prob_flip, [3])

                # Average TTA
                avg_prob = (prob + prob_flip) / 2.0
                batch_preds.append(avg_prob)

            # Average Ensemble
            final_prob = torch.stack(batch_preds).mean(dim=0)

            # Unpad
            final_prob = final_prob[:, 0, 13:114, 13:114]

            # Binarize
            pred_masks = (final_prob > 0.5).byte().cpu().numpy()

            # RLE Encode
            start_idx = i * 32
            for j, mask in enumerate(pred_masks):
                rle = rle_encode(mask)
                rle_list.append(rle)
                id_list.append(test_ids[start_idx + j])

    # Save Submission
    sub_df = pd.DataFrame({"id": id_list, "rle_mask": rle_list})
    os.makedirs("submission", exist_ok=True)
    sub_df.to_csv("submission/submission.csv", index=False)
    print("Submission saved to submission/submission.csv")


def main():
    train_model(epochs=150, batch_size=32, debug=False)
    predict_and_submit()


if __name__ == "__main__":
    main()

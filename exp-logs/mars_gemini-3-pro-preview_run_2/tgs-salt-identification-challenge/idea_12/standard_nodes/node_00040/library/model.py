import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import glob

# Import from provided libraries
from library.utils import set_seed, rle_encode, calculate_iou_map
from library.losses import CombinedLoss

# ==========================================
# Constants & Configuration
# ==========================================
IDEA_DIR = "./working/idea_12"
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
IMG_SIZE = 128
ORIG_SIZE = 101
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# Model Architecture
# ==========================================


class DepthProjector(nn.Module):
    """
    Projects scalar depth into a high-dimensional embedding using an MLP.
    Cite solution_lesson_node_00029: Non-linear embeddings outperform linear projections.
    """

    def __init__(self, input_dim=1, output_dim=32):
        super(DepthProjector, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(output_dim, output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class WideDecoderBlock(nn.Module):
    """
    LinkNet-style decoder block with width correction.
    Structure: 1x1 Conv (reduce) -> 3x3 Deconv (upsample) -> 1x1 Conv (expand)
    Internal width is in_channels // 4.
    """

    def __init__(self, in_channels, out_channels):
        super(WideDecoderBlock, self).__init__()
        internal_dim = in_channels // 4

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, internal_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(internal_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(
                internal_dim,
                internal_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(internal_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(internal_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DepthAwareLinkNet34(nn.Module):
    def __init__(self):
        super(DepthAwareLinkNet34, self).__init__()

        # Load Pretrained ResNet34
        base = models.resnet34(pretrained=True)

        # Input Adaptation: Modify first conv for 1-channel input
        w = base.conv1.weight
        base.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Sum weights across RGB channels
        base.conv1.weight.data = w.sum(dim=1, keepdim=True)

        # Encoder Layers
        self.in_block = nn.Sequential(base.conv1, base.bn1, base.relu)
        self.maxpool = base.maxpool
        self.enc1 = base.layer1  # 64 channels, 32x32
        self.enc2 = base.layer2  # 128 channels, 16x16
        self.enc3 = base.layer3  # 256 channels, 8x8
        self.enc4 = base.layer4  # 512 channels, 4x4

        # Depth Injection
        # Cite solution_lesson_node_00037: Additive concatenation > Multiplicative FiLM
        self.depth_projector = DepthProjector(1, 32)

        # Decoder Layers (Wide-LinkNet)
        # Dec4: 512 (encoder) + 32 (depth) = 544 -> 256
        # Cite solution_lesson_node_00019: Remove redundant bottleneck, pass full features
        self.dec4 = WideDecoderBlock(544, 256)
        # Dec3: 256 -> 128
        self.dec3 = WideDecoderBlock(256, 128)
        # Dec2: 128 -> 64
        self.dec2 = WideDecoderBlock(128, 64)
        # Dec1: 64 -> 64
        self.dec1 = WideDecoderBlock(64, 64)

        # Final Upsample: 64x64 -> 128x128
        self.final_dec = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x, depth):
        # x: (B, 1, 128, 128)
        # depth: (B, 1)

        # Encoder
        x0 = self.in_block(x)  # (B, 64, 64, 64)
        x1 = self.maxpool(x0)  # (B, 64, 32, 32)
        e1 = self.enc1(x1)  # (B, 64, 32, 32)
        e2 = self.enc2(e1)  # (B, 128, 16, 16)
        e3 = self.enc3(e2)  # (B, 256, 8, 8)
        e4 = self.enc4(e3)  # (B, 512, 4, 4)

        # Bottleneck: Concatenation
        d = self.depth_projector(depth)  # (B, 32)
        d = d.unsqueeze(2).unsqueeze(3).expand(-1, -1, e4.size(2), e4.size(3))
        b = torch.cat([e4, d], dim=1)  # (B, 544, 4, 4)

        # Decoder with Additive Skips
        d4 = self.dec4(b)  # (B, 256, 8, 8)
        d4 = d4 + e3

        d3 = self.dec3(d4)  # (B, 128, 16, 16)
        d3 = d3 + e2

        d2 = self.dec2(d3)  # (B, 64, 32, 32)
        d2 = d2 + e1

        d1 = self.dec1(d2)  # (B, 64, 64, 64)
        d1 = d1 + x0

        out = self.final_dec(d1)  # (B, 32, 128, 128)
        logits = self.final_conv(out)

        return logits


# ==========================================
# Data Processing & Caching
# ==========================================


def load_and_cache_data(load_cached=True):
    """
    Loads data from disk or cache.
    Strictly follows the requirement: Check cache -> If missing/forced, compute & save.
    """
    os.makedirs(IDEA_DIR, exist_ok=True)

    cache_files = {
        "train_images": os.path.join(IDEA_DIR, "train_images.npy"),
        "train_masks": os.path.join(IDEA_DIR, "train_masks.npy"),
        "train_depths": os.path.join(IDEA_DIR, "train_depths.npy"),
        "val_images": os.path.join(IDEA_DIR, "val_images.npy"),
        "val_masks": os.path.join(IDEA_DIR, "val_masks.npy"),
        "val_depths": os.path.join(IDEA_DIR, "val_depths.npy"),
        "test_images": os.path.join(IDEA_DIR, "test_images.npy"),
        "test_depths": os.path.join(IDEA_DIR, "test_depths.npy"),
        "test_ids": os.path.join(IDEA_DIR, "test_ids.npy"),  # Save IDs for submission
    }

    all_exist = all(os.path.exists(f) for f in cache_files.values())

    if load_cached and all_exist:
        print("Loading data from cache...")
        data = {k: np.load(v, allow_pickle=True) for k, v in cache_files.items()}
        return data

    print("Cache missing or reload requested. Processing data from scratch...")

    # Helper to load images
    def load_set(df, is_test=False):
        images = []
        masks = []
        depths = []
        ids = []

        for idx, row in df.iterrows():
            # Load Image
            img_path = os.path.join(INPUT_DIR, row["image_path"])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            images.append(img)

            # Load Depth
            depths.append(row["z"])
            ids.append(row["id"])

            if not is_test:
                # Load Mask
                mask_path = os.path.join(INPUT_DIR, row["mask_path"])
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                # Binarize (just in case)
                mask = (mask > 127).astype(np.uint8)
                masks.append(mask)

        return (
            np.array(images),
            np.array(masks) if not is_test else None,
            np.array(depths),
            np.array(ids),
        )

    # Load Metadata
    train_df = pd.read_csv(os.path.join(METADATA_DIR, "train.csv"))
    val_df = pd.read_csv(os.path.join(METADATA_DIR, "val.csv"))
    test_df = pd.read_csv(os.path.join(METADATA_DIR, "test.csv"))

    # Process
    train_imgs, train_msks, train_d, _ = load_set(train_df)
    val_imgs, val_msks, val_d, _ = load_set(val_df)
    test_imgs, _, test_d, test_ids = load_set(test_df, is_test=True)

    # Save to cache
    np.save(cache_files["train_images"], train_imgs)
    np.save(cache_files["train_masks"], train_msks)
    np.save(cache_files["train_depths"], train_d)
    np.save(cache_files["val_images"], val_imgs)
    np.save(cache_files["val_masks"], val_msks)
    np.save(cache_files["val_depths"], val_d)
    np.save(cache_files["test_images"], test_imgs)
    np.save(cache_files["test_depths"], test_d)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "train_images": train_imgs,
        "train_masks": train_msks,
        "train_depths": train_d,
        "val_images": val_imgs,
        "val_masks": val_msks,
        "val_depths": val_d,
        "test_images": test_imgs,
        "test_depths": test_d,
        "test_ids": test_ids,
    }


class SaltDataset(Dataset):
    def __init__(
        self,
        images,
        masks,
        depths,
        transform=None,
        depth_mean=0,
        depth_std=1,
        training=True,
    ):
        self.images = images
        self.masks = masks
        self.depths = depths
        self.transform = transform
        self.depth_mean = depth_mean
        self.depth_std = depth_std
        self.training = training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        depth = self.depths[idx]

        # Normalize depth
        z = (depth - self.depth_mean) / self.depth_std

        # Bernoulli Depth Masking (Train only)
        # p=0.5 to set depth to 0 (mean)
        if self.training and np.random.rand() < 0.5:
            z = 0.0

        # If testing, we force z=0 as per strategy
        if not self.training and self.masks is None:
            z = 0.0

        if self.masks is not None:
            mask = self.masks[idx]
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image = augmented["image"]
                mask = augmented["mask"]
            return (
                image,
                mask.unsqueeze(0).float(),
                torch.tensor([z], dtype=torch.float32),
            )
        else:
            if self.transform:
                augmented = self.transform(image=image)
                image = augmented["image"]
            return image, torch.tensor([z], dtype=torch.float32)


def get_transforms(phase):
    # Padding to 128x128
    pad_h = IMG_SIZE - ORIG_SIZE
    pad_w = IMG_SIZE - ORIG_SIZE
    # PadIfNeeded centers by default if result is larger than image

    if phase == "train":
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.ElasticTransform(
                    alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.2
                ),
                A.ShiftScaleRotate(
                    shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.2
                ),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                    mean=(0.485,), std=(0.229,)
                ),  # ImageNet mean/std for 1 channel?
                # Actually ImageNet is RGB. We summed weights.
                # Let's use the dataset stats calculated in EDA: Mean ~148/255=0.58, Std ~65/255=0.25
                # Or just standard 0.5, 0.5.
                # Using standard ImageNet stats on grayscale is common practice when using pretrained weights.
                # (0.485+0.456+0.406)/3 = 0.449. Let's use 0.45, 0.225
                # Or just use the values provided in albumentations for grayscale.
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.PadIfNeeded(
                    min_height=IMG_SIZE,
                    min_width=IMG_SIZE,
                    border_mode=cv2.BORDER_REFLECT,
                ),
                A.Normalize(mean=(0.485,), std=(0.229,)),
                ToTensorV2(),
            ]
        )


# ==========================================
# Training & Evaluation
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, depths in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()
        outputs = model(images, depths)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, device):
    model.eval()
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths in loader:
            images = images.to(device)
            depths = depths.to(device)

            outputs = torch.sigmoid(model(images, depths))

            # Crop back to 101x101 for metric calculation
            # Center crop
            h, w = outputs.shape[2], outputs.shape[3]
            start_h = (h - ORIG_SIZE) // 2
            start_w = (w - ORIG_SIZE) // 2
            outputs = outputs[
                :, :, start_h : start_h + ORIG_SIZE, start_w : start_w + ORIG_SIZE
            ]

            all_preds.append(outputs.cpu().numpy())
            all_masks.append(
                masks.cpu().numpy()
            )  # Masks are already original size? No, transformed.
            # Wait, validation transform also pads masks. We must crop masks too.

    all_preds = np.concatenate(all_preds)
    all_masks = np.concatenate(all_masks)

    # Crop masks
    h, w = all_masks.shape[2], all_masks.shape[3]
    start_h = (h - ORIG_SIZE) // 2
    start_w = (w - ORIG_SIZE) // 2
    all_masks = all_masks[
        :, :, start_h : start_h + ORIG_SIZE, start_w : start_w + ORIG_SIZE
    ]

    # Linear search for best threshold
    best_threshold = 0.5
    best_score = -1

    for t in np.arange(0.3, 0.7, 0.05):
        score = calculate_iou_map(all_preds, all_masks, threshold=t)
        if score > best_score:
            best_score = score
            best_threshold = t

    return best_score, best_threshold


def run_training(epochs=50, batch_size=32, lr=1e-4):
    set_seed(42)

    # Load Data
    data = load_and_cache_data(load_cached=True)

    # Calc depth stats
    all_depths = np.concatenate([data["train_depths"], data["val_depths"]])
    d_mean = all_depths.mean()
    d_std = all_depths.std()

    train_dataset = SaltDataset(
        data["train_images"],
        data["train_masks"],
        data["train_depths"],
        transform=get_transforms("train"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=True,
    )
    val_dataset = SaltDataset(
        data["val_images"],
        data["val_masks"],
        data["val_depths"],
        transform=get_transforms("val"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = ResFiLM_LinkNet34().to(DEVICE)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_map = 0.0
    best_thresh = 0.5
    patience = 10
    no_improve = 0

    print(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_map, thresh = validate(model, val_loader, DEVICE)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {train_loss:.4f} | Val mAP: {val_map:.6f} | Thresh: {thresh:.2f}"
        )

        if val_map > best_map:
            best_map = val_map
            best_thresh = thresh
            torch.save(model.state_dict(), os.path.join(IDEA_DIR, "best_model.pth"))
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            print("Early stopping triggered.")
            break

    print(f"Best Validation mAP: {best_map:.6f} at Threshold: {best_thresh:.2f}")
    return best_thresh, d_mean, d_std


def generate_submission(best_threshold, d_mean, d_std):
    print("Generating submission...")
    data = load_and_cache_data(load_cached=True)

    test_dataset = SaltDataset(
        data["test_images"],
        None,
        data["test_depths"],
        transform=get_transforms("test"),
        depth_mean=d_mean,
        depth_std=d_std,
        training=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = ResFiLM_LinkNet34().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(IDEA_DIR, "best_model.pth")))
    model.eval()

    rles = []
    ids = data["test_ids"]

    with torch.no_grad():
        for images, depths in test_loader:
            images = images.to(DEVICE)
            depths = depths.to(DEVICE)

            # TTA: Original + Horizontal Flip
            # Original
            out1 = torch.sigmoid(model(images, depths))

            # Flip
            images_flipped = torch.flip(images, dims=[3])
            out2 = torch.sigmoid(model(images_flipped, depths))
            out2 = torch.flip(out2, dims=[3])

            # Average
            preds = (out1 + out2) / 2.0

            # Crop
            h, w = preds.shape[2], preds.shape[3]
            start_h = (h - ORIG_SIZE) // 2
            start_w = (w - ORIG_SIZE) // 2
            preds = preds[
                :, 0, start_h : start_h + ORIG_SIZE, start_w : start_w + ORIG_SIZE
            ]

            preds = (preds > best_threshold).cpu().numpy().astype(np.uint8)

            for p in preds:
                rles.append(rle_encode(p))

    sub_df = pd.DataFrame({"id": ids, "rle_mask": rles})
    os.makedirs("submission", exist_ok=True)
    sub_df.to_csv("submission/submission.csv", index=False)
    print("Submission saved to submission/submission.csv")


def run_task():
    thresh, dm, ds = run_training(epochs=50)
    generate_submission(thresh, dm, ds)


# Note: No if __name__ == "__main__" block as requested.
# The user can import run_task and execute it.

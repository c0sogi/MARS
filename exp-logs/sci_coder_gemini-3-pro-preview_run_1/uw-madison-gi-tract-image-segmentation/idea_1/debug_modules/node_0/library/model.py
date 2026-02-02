import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
from library.utils import set_seed, dice_coefficient, hausdorff_distance, rle_encode
from library.dataset import UWMadisonDataset

# --- Configuration & Constants ---
WORKING_DIR = "./working/idea_1"
SUBMISSION_DIR = "./submission"
CHECKPOINT_PATH = os.path.join(WORKING_DIR, "best_model.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["large_bowel", "small_bowel", "stomach"]

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# --- Model Architecture ---


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(DecoderBlock, self).__init__()
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

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle padding issues if shapes don't match exactly due to odd dimensions
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(
                x, size=skip.shape[2:], mode="bilinear", align_corners=True
            )
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetResNet18(nn.Module):
    def __init__(self, num_classes=3):
        super(UNetResNet18, self).__init__()

        # Encoder: ResNet18
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first layer for 1-channel input
        # Sum weights across RGB channels to keep intensity magnitude roughly similar
        original_weights = base_model.conv1.weight.data
        new_weights = original_weights.sum(dim=1, keepdim=True)
        base_model.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        base_model.conv1.weight.data = new_weights

        self.encoder0 = nn.Sequential(
            base_model.conv1, base_model.bn1, base_model.relu
        )  # Output: 64 ch, 1/2 size
        self.pool = base_model.maxpool  # Output: 64 ch, 1/4 size

        self.encoder1 = base_model.layer1  # 64 ch, 1/4 size
        self.encoder2 = base_model.layer2  # 128 ch, 1/8 size
        self.encoder3 = base_model.layer3  # 256 ch, 1/16 size
        self.encoder4 = base_model.layer4  # 512 ch, 1/32 size

        # Decoder
        # Layer 4 (512) -> Layer 3 (256)
        self.decoder4 = DecoderBlock(512, 256, 256)
        # Layer 3 (256) -> Layer 2 (128)
        self.decoder3 = DecoderBlock(256, 128, 128)
        # Layer 2 (128) -> Layer 1 (64)
        self.decoder2 = DecoderBlock(128, 64, 64)
        # Layer 1 (64) -> Layer 0 (64)
        self.decoder1 = DecoderBlock(64, 64, 32)

        # Final upsample to original size
        self.final_upsample = nn.Upsample(
            scale_factor=2, mode="bilinear", align_corners=True
        )
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, kernel_size=1),
        )

    def forward(self, x):
        # Encoder
        x0 = self.encoder0(x)  # 1/2
        p0 = self.pool(x0)  # 1/4
        x1 = self.encoder1(p0)  # 1/4
        x2 = self.encoder2(x1)  # 1/8
        x3 = self.encoder3(x2)  # 1/16
        x4 = self.encoder4(x3)  # 1/32

        # Decoder
        d4 = self.decoder4(x4, x3)  # 1/16
        d3 = self.decoder3(d4, x2)  # 1/8
        d2 = self.decoder2(d3, x1)  # 1/4
        d1 = self.decoder1(d2, x0)  # 1/2

        out = self.final_upsample(d1)  # 1/1
        out = self.final_conv(out)

        return torch.sigmoid(out)


# --- Loss Function ---


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce = self.bce(pred, target)

        # Dice Loss
        smooth = 1e-6
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice_score = (2.0 * intersection + smooth) / (
            pred_flat.sum() + target_flat.sum() + smooth
        )
        dice_loss = 1 - dice_score

        return self.bce_weight * bce + (1 - self.bce_weight) * dice_loss


# --- Training Logic ---


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    dice_scores = []
    hausdorff_scores = []

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * images.size(0)

            # Metrics calculation per sample in batch
            # Threshold predictions
            preds = (outputs > 0.5).float()

            for i in range(images.size(0)):
                # Calculate metrics for each class and average
                d_score = 0
                h_score = 0
                for c in range(3):
                    d_score += dice_coefficient(masks[i, c], preds[i, c])
                    h_score += hausdorff_distance(masks[i, c], preds[i, c])

                dice_scores.append(d_score / 3.0)
                hausdorff_scores.append(h_score / 3.0)

    epoch_loss = running_loss / len(loader.dataset)
    mean_dice = np.mean(dice_scores)
    mean_hausdorff = np.mean(hausdorff_scores)

    # Competition metric: 0.4 * Dice + 0.6 * Hausdorff (Normalized 0-1, where 1 is bad for distance?
    # Actually HD is distance, so lower is better. But prompt says "0-1 score".
    # Usually HD score is 1 - norm_dist or similar if higher is better.
    # However, standard HD is a distance (lower is better).
    # The prompt says: "The two metrics are combined...". Usually in this comp it was 0.4*Dice + 0.6*(1-HD).
    # Given the ambiguity, I will track both raw values.
    # To select best model, I'll maximize Dice for simplicity as it correlates well with segmentation quality.

    return epoch_loss, mean_dice, mean_hausdorff


def train_model(
    epochs=10, batch_size=32, lr=1e-4, fraction=1.0, img_size=256, patience=5
):
    set_seed(42)

    # Datasets
    train_dataset = UWMadisonDataset(mode="train", fraction=fraction, img_size=img_size)
    val_dataset = UWMadisonDataset(mode="val", fraction=fraction, img_size=img_size)

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

    print(
        f"Training on {len(train_dataset)} samples, Validating on {len(val_dataset)} samples."
    )

    # Model & Training components
    model = UNetResNet18(num_classes=3).to(DEVICE)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    best_dice = 0.0
    epochs_no_improve = 0

    for epoch in range(epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_dice, val_hd = validate(model, val_loader, criterion, DEVICE)

        # Scheduler step
        scheduler.step(val_dice)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{epochs} | Time: {duration:.0f}s | "
            f"Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | "
            f"Val Dice: {val_dice:.5f} | Val HD: {val_hd:.5f}"
        )

        # Checkpointing
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Early Stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Best Validation Dice: {best_dice:.5f}")
    return model


# --- Inference Logic ---


def predict_and_submit(batch_size=32, img_size=256):
    print("Starting Inference...")

    # Load Model
    model = UNetResNet18(num_classes=3).to(DEVICE)
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Loaded best model checkpoint.")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()

    # Test Dataset
    test_dataset = UWMadisonDataset(mode="test", img_size=img_size)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    results = []

    with torch.no_grad():
        for images, ids, original_shapes in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = (outputs > 0.5).float().cpu().numpy()

            # Iterate through batch
            for i in range(len(ids)):
                case_id = ids[i]
                orig_h, orig_w = original_shapes[i]

                # Resize prediction back to original size
                # Preds is (3, 256, 256) -> need (orig_h, orig_w, 3) for processing or keep channel first
                # Using cv2 for resizing masks is standard

                for class_idx, class_name in enumerate(CLASSES):
                    mask = preds[i, class_idx]  # (256, 256)

                    # Resize to original resolution
                    if (mask.shape[0] != orig_h) or (mask.shape[1] != orig_w):
                        # cv2.resize expects (W, H)
                        mask = cv2_resize_mask(mask, (orig_w, orig_h))

                    # Binarize again after resize interpolation
                    mask = (mask > 0.5).astype(np.uint8)

                    rle = rle_encode(mask)
                    results.append(
                        {"id": case_id, "class": class_name, "predicted": rle}
                    )

    # Create Submission DataFrame
    submission_df = pd.DataFrame(results)

    # Ensure all columns are present and format is correct
    if submission_df.empty:
        # Create empty submission if no data (should not happen)
        submission_df = pd.DataFrame(columns=["id", "class", "predicted"])

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    submission_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")


def cv2_resize_mask(mask, target_shape):
    # target_shape is (W, H)
    import cv2

    return cv2.resize(mask, target_shape, interpolation=cv2.INTER_LINEAR)


# --- Main Interface ---


def run_training(epochs=15, batch_size=32, fraction=1.0):
    train_model(epochs=epochs, batch_size=batch_size, fraction=fraction)


def run_inference():
    predict_and_submit()

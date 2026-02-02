import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

# Import from provided libraries
from library.dataset import get_dataloaders, get_test_loader, ORIG_SIZE, TARGET_SIZE
from library.utils import set_seed, rle_encode, do_kaggle_metric

# -------------------------------------------------------------------------
# 1. Architecture Components
# -------------------------------------------------------------------------


class SCSEModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        # Channel Squeeze and Excitation (cSE)
        # Global Average Pooling -> Dense -> ReLU -> Dense -> Sigmoid
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, max(1, in_channels // reduction), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, in_channels // reduction), in_channels, 1),
            nn.Sigmoid(),
        )
        # Spatial Squeeze and Excitation (sSE)
        # Conv 1x1 -> Sigmoid
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.scse = SCSEModule(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.scse(x)
        return x


class DepthAwareUNet(nn.Module):
    def __init__(self, in_channels=2, n_classes=1):
        super().__init__()

        # Encoder
        self.conv1 = ConvBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)

        # Center
        self.center = ConvBlock(256, 512)

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec4 = ConvBlock(512 + 256, 256)

        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec3 = ConvBlock(256 + 128, 128)

        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec2 = ConvBlock(128 + 64, 64)

        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.dec1 = ConvBlock(64 + 32, 32)

        self.final = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)
        pool4 = self.pool4(conv4)

        # Center
        center = self.center(pool4)

        # Decoder
        up4 = self.up4(center)
        merge4 = torch.cat([up4, conv4], dim=1)
        dec4 = self.dec4(merge4)

        up3 = self.up3(dec4)
        merge3 = torch.cat([up3, conv3], dim=1)
        dec3 = self.dec3(merge3)

        up2 = self.up2(dec3)
        merge2 = torch.cat([up2, conv2], dim=1)
        dec2 = self.dec2(merge2)

        up1 = self.up1(dec2)
        merge1 = torch.cat([up1, conv1], dim=1)
        dec1 = self.dec1(merge1)

        return self.final(dec1)


# -------------------------------------------------------------------------
# 2. Training Utilities
# -------------------------------------------------------------------------


class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: logits, targets: binary

        # BCE
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="mean")

        # Dice
        inputs_sigmoid = torch.sigmoid(inputs)

        # Flatten
        inputs_flat = inputs_sigmoid.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice = 1 - (2.0 * intersection + smooth) / (
            inputs_flat.sum() + targets_flat.sum() + smooth
        )

        return bce + dice


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, masks in loader:
        inputs = inputs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for inputs, masks in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, masks)
            running_loss += loss.item() * inputs.size(0)

            # Collect for metric
            preds_prob = torch.sigmoid(outputs)
            all_preds.append(preds_prob.cpu().numpy())
            all_masks.append(masks.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate Metric
    all_preds = np.concatenate(all_preds, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    # Center crop back to 101x101 before metric
    # Current: 128x128
    pad_h = TARGET_SIZE - ORIG_SIZE
    pad_top = pad_h // 2
    pad_w = TARGET_SIZE - ORIG_SIZE
    pad_left = pad_w // 2

    preds_cropped = all_preds[
        :, :, pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
    ]
    masks_cropped = all_masks[
        :, :, pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
    ]

    score = do_kaggle_metric(preds_cropped, masks_cropped, threshold=0.5)

    return epoch_loss, score


def predict_and_submit(model, device):
    model.eval()
    test_loader = get_test_loader(batch_size=32)

    pad_h = TARGET_SIZE - ORIG_SIZE
    pad_top = pad_h // 2
    pad_w = TARGET_SIZE - ORIG_SIZE
    pad_left = pad_w // 2

    all_rles = []
    ids = test_loader.dataset.ids
    current_idx = 0

    with torch.no_grad():
        for inputs, _ in test_loader:
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # TTA: Original
            out = model(inputs)
            prob = torch.sigmoid(out)

            # TTA: Flip
            inputs_flip = torch.flip(inputs, dims=[3])
            out_flip = model(inputs_flip)
            prob_flip = torch.flip(torch.sigmoid(out_flip), dims=[3])

            # Average
            avg_prob = (prob + prob_flip) / 2.0

            # Crop
            avg_prob = avg_prob[
                :, 0, pad_top : pad_top + ORIG_SIZE, pad_left : pad_left + ORIG_SIZE
            ]

            # Threshold
            preds = (avg_prob > 0.5).cpu().numpy().astype(np.uint8)

            for i in range(batch_size):
                rle = rle_encode(preds[i])
                img_id = ids[current_idx]
                all_rles.append(f"{img_id},{rle}")
                current_idx += 1

    # Save
    os.makedirs("submission", exist_ok=True)
    with open("submission/submission.csv", "w") as f:
        f.write("id,rle_mask\n")
        f.write("\n".join(all_rles))

    print("Submission saved to submission/submission.csv")


def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters
    BATCH_SIZE = 32
    EPOCHS = 50
    LR = 1e-4

    # Data
    train_loader, val_loader = get_dataloaders(batch_size=BATCH_SIZE)

    # Model
    model = ResNet34UNetPlusPlus(in_channels=2, n_classes=1).to(device)

    # Training Setup
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_loss = float("inf")

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_score = validate(model, val_loader, criterion, device)

        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Score: {val_score:.6f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            # Save best model
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/best_model.pth")

    print(f"Best Val Loss: {best_loss:.6f}")

    # Load best
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))

    # Predict
    predict_and_submit(model, device)

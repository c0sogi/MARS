import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
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


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

        self.scse = SCSEModule(out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        x = self.scse(x)
        return x


class DepthAwareUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(DepthAwareUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)

        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = nn.Conv2d(16, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


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

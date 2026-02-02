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


class ResNet34UNetPlusPlus(nn.Module):
    def __init__(self, in_channels=2, n_classes=1):
        super().__init__()

        # Load Pretrained ResNet34
        resnet = models.resnet34(pretrained=True)

        # ---------------------------------------------------------------------
        # Encoder Modification for 128x128 input and 2 channels
        # ---------------------------------------------------------------------
        # Original conv1: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We change to stride 1 to keep resolution high (128x128)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        # We remove maxpool to preserve spatial dimensions at the start
        # self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1  # 64ch, 128x128 (if no maxpool and conv1 stride 1)
        self.layer2 = resnet.layer2  # 128ch, 64x64
        self.layer3 = resnet.layer3  # 256ch, 32x32
        self.layer4 = resnet.layer4  # 512ch, 16x16

        # ---------------------------------------------------------------------
        # Decoder (U-Net++)
        # ---------------------------------------------------------------------
        # Filters for each level: 0->64, 1->128, 2->256, 3->512
        filters = [64, 128, 256, 512]

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # j=1 (Standard U-Net links)
        # X_2_1: Up(X_3_0) + X_2_0 -> 512 + 256 -> 256
        self.conv2_1 = ConvBlock(filters[3] + filters[2], filters[2])
        # X_1_1: Up(X_2_0) + X_1_0 -> 256 + 128 -> 128
        self.conv1_1 = ConvBlock(filters[2] + filters[1], filters[1])
        # X_0_1: Up(X_1_0) + X_0_0 -> 128 + 64 -> 64
        self.conv0_1 = ConvBlock(filters[1] + filters[0], filters[0])

        # j=2
        # X_1_2: Up(X_2_1) + X_1_0 + X_1_1 -> 256 + 128 + 128 -> 128
        self.conv1_2 = ConvBlock(filters[2] + filters[1] * 2, filters[1])
        # X_0_2: Up(X_1_1) + X_0_0 + X_0_1 -> 128 + 64 + 64 -> 64
        self.conv0_2 = ConvBlock(filters[1] + filters[0] * 2, filters[0])

        # j=3
        # X_0_3: Up(X_1_2) + X_0_0 + X_0_1 + X_0_2 -> 128 + 64*3 -> 64
        self.conv0_3 = ConvBlock(filters[1] + filters[0] * 3, filters[0])

        # Final Output
        self.final_conv = nn.Conv2d(filters[0], n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0_0 = self.relu(self.bn1(self.conv1(x)))  # 128x128
        x0_0 = self.layer1(x0_0)  # 128x128 (ResNet layer1 usually doesn't downsample)

        x1_0 = self.layer2(x0_0)  # 64x64
        x2_0 = self.layer3(x1_0)  # 32x32
        x3_0 = self.layer4(x2_0)  # 16x16

        # Decoder j=1
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        # Decoder j=2
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        # Decoder j=3
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        out = self.final_conv(x0_3)
        return out


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

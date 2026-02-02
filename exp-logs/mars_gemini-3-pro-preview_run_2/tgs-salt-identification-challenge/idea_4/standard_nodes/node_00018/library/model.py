import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torch.optim import lr_scheduler

from library.config import Config
from library.utils import do_kaggle_metric, rle_encode, unpad_image
from library.loss import BCELovaszLoss
from library.dataset import get_dataloaders

# ==================================================================================
# Model Architecture
# ==================================================================================


class LinkNetBlock(nn.Module):
    """
    LinkNet Decoder Block.
    Efficient upsampling block using 1x1 convs to reduce dimensions before transposed conv.
    """

    def __init__(self, in_channels, out_channels):
        super(LinkNetBlock, self).__init__()

        # Internal width is typically in_channels // 4 for efficiency
        inter_channels = in_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                inter_channels,
                inter_channels,
                3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(inter_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.deconv(x)
        x = self.conv2(x)
        return x


class SaltNet(nn.Module):
    """
    ResNet34-LinkNet with ASPP Bottleneck and Depth Injection.
    """

    def __init__(self):
        super(SaltNet, self).__init__()

        # 1. Encoder: ResNet34
        # Load pretrained weights
        m = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)

        # Modify first layer for 1-channel input (Grayscale)
        # We sum the weights of the 3 RGB channels to preserve intensity information
        original_conv1 = m.conv1
        m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            m.conv1.weight.data = original_conv1.weight.data.sum(dim=1, keepdim=True)

        self.encoder0 = nn.Sequential(m.conv1, m.bn1, m.relu)  # Out: 64x64, 64ch
        self.encoder1 = m.maxpool  # Out: 32x32, 64ch
        self.encoder_layer1 = m.layer1  # Out: 32x32, 64ch (Stride 1)
        self.encoder_layer2 = m.layer2  # Out: 16x16, 128ch
        self.encoder_layer3 = m.layer3  # Out: 8x8, 256ch
        self.encoder_layer4 = m.layer4  # Out: 4x4, 512ch

        # 2. Bottleneck
        # Cite solution_lesson_node_00017: Replace ASPP with simple convolution for small feature maps
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        # 3. Depth Injection
        # Projects scalar depth to a vector
        self.depth_embedding = nn.Sequential(
            nn.Linear(1, Config.DEPTH_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(Config.DEPTH_EMBEDDING_DIM, Config.DEPTH_EMBEDDING_DIM),
            nn.ReLU(inplace=True),
        )

        # 4. Decoder: LinkNet Style
        # Input to decoder is ASPP(256) + Depth(32) = 288 channels

        # Block 4: 4x4 -> 8x8. Skip: Layer3 (256ch).
        self.decoder4 = LinkNetBlock(256 + Config.DEPTH_EMBEDDING_DIM, 256)

        # Block 3: 8x8 -> 16x16. Skip: Layer2 (128ch).
        self.decoder3 = LinkNetBlock(256, 128)

        # Block 2: 16x16 -> 32x32. Skip: Layer1 (64ch).
        self.decoder2 = LinkNetBlock(128, 64)

        # Block 1: 32x32 -> 64x64. Skip: Encoder0 (64ch).
        self.decoder1 = LinkNetBlock(64, 64)

        # 5. Final Head
        # Upsample 64x64 -> 128x128
        self.final_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, 3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),  # Logits
        )

    def forward(self, x, depth):
        # Encoder
        e0 = self.encoder0(x)  # 64x64, 64
        e1 = self.encoder1(e0)  # 32x32, 64
        e1 = self.encoder_layer1(e1)  # 32x32, 64
        e2 = self.encoder_layer2(e1)  # 16x16, 128
        e3 = self.encoder_layer3(e2)  # 8x8, 256
        e4 = self.encoder_layer4(e3)  # 4x4, 512

        # Bottleneck
        b = self.bottleneck(e4)  # 4x4, 256

        # Depth Injection
        d = self.depth_embedding(depth)  # (N, 32)
        d = d.unsqueeze(2).unsqueeze(3)  # (N, 32, 1, 1)
        d = d.expand(-1, -1, b.size(2), b.size(3))  # (N, 32, 4, 4)

        b = torch.cat([b, d], dim=1)  # (N, 288, 4, 4)

        # Decoder
        d4 = self.decoder4(b)  # 8x8, 256
        d4 = d4 + e3  # Add Skip (256)

        d3 = self.decoder3(d4)  # 16x16, 128
        d3 = d3 + e2  # Add Skip (128)

        d2 = self.decoder2(d3)  # 32x32, 64
        d2 = d2 + e1  # Add Skip (64)

        d1 = self.decoder1(d2)  # 64x64, 64
        d1 = d1 + e0  # Add Skip (64)

        # Final
        out = self.final_deconv(d1)  # 128x128, 1

        return out


# ==================================================================================
# Training & Inference Logic
# ==================================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, depths, ids in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        logits = model(images, depths)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_masks = []

    with torch.no_grad():
        for images, masks, depths, ids in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            logits = model(images, depths)
            loss = criterion(logits, masks)

            running_loss += loss.item() * images.size(0)

            # Sigmoid for metric calculation
            preds = torch.sigmoid(logits).cpu().numpy()

            # Unpad predictions and masks for accurate metric calculation
            for i in range(len(preds)):
                p = unpad_image(preds[i, 0], Config.ORIG_SIZE)
                m = unpad_image(masks[i, 0].cpu().numpy(), Config.ORIG_SIZE)
                all_preds.append(p)
                all_masks.append(m)

    epoch_loss = running_loss / len(loader.dataset)

    # Calculate MAP
    all_preds = np.array(all_preds)
    all_masks = np.array(all_masks)
    map_score = do_kaggle_metric(all_preds, all_masks, threshold=0.5)

    return epoch_loss, map_score


def train_model(config=Config):
    device = config.DEVICE
    print(f"Training on device: {device}")

    # Data
    train_loader, val_loader, _ = get_dataloaders(config)

    # Model
    model = SaltNet().to(device)

    # Loss & Optimizer
    criterion = BCELovaszLoss(
        bce_weight=config.BCE_WEIGHT, lovasz_weight=config.LOVASZ_WEIGHT
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
    )

    # Loop
    best_map = 0.0
    patience_counter = 0

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_map = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val MAP: {val_map:.6f} | "
            f"Time: {time.time() - start_time:.1f}s"
        )

        # Save Best
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  >>> New Best Model Saved! MAP: {best_map:.6f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best MAP: {best_map:.6f}")


def predict(config=Config):
    device = config.DEVICE
    print("Starting Inference...")

    # Data
    _, _, test_loader = get_dataloaders(config)

    # Model
    model = SaltNet().to(device)
    if os.path.exists(config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
    else:
        print("Warning: Checkpoint not found. Using untrained model.")

    model.eval()

    predictions = {}

    with torch.no_grad():
        for images, depths, ids in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Original
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # TTA: Flip
            if config.TTA_FLIP:
                images_flip = torch.flip(images, dims=[3])
                logits_flip = model(images_flip, depths)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip = torch.flip(probs_flip, dims=[3])
                probs = (probs + probs_flip) / 2.0

            probs = probs.cpu().numpy()

            for i, img_id in enumerate(ids):
                # Unpad
                pred_mask = unpad_image(probs[i, 0], Config.ORIG_SIZE)

                # Binarize
                pred_bin = (pred_mask > 0.5).astype(np.uint8)

                # RLE Encode
                rle = rle_encode(pred_bin)
                predictions[img_id] = rle

    # Save Submission
    sub_df = pd.DataFrame.from_dict(predictions, orient="index", columns=["rle_mask"])
    sub_df.index.name = "id"
    sub_df.reset_index(inplace=True)
    sub_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def run_pipeline():
    """
    Main entry point to run training and prediction.
    """
    # Ensure config dirs exist
    Config.create_dirs()

    # Train
    train_model(Config)

    # Predict
    predict(Config)

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.models import resnet34, ResNet34_Weights
from tqdm import tqdm

from library import config, dataset, losses, utils

# ==========================================
# Model Architecture
# ==========================================


class DecoderBlock(nn.Module):
    """
    Wide-LinkNet Decoder Block with Additive Skip Connections.
    Internal dimension is calculated as in_channels // 4 to preserve information.
    """

    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()

        # "Wide" internal width
        mid_channels = in_channels // 4

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.trans = nn.Sequential(
            nn.ConvTranspose2d(
                mid_channels,
                mid_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip=None):
        out = self.conv1(x)
        out = self.trans(out)
        out = self.conv2(out)

        if skip is not None:
            out = out + skip

        return out


class WideLinkNet34(nn.Module):
    """
    Multi-Task Wide-LinkNet34 with Auxiliary Depth Regression.
    """

    def __init__(self):
        super(WideLinkNet34, self).__init__()

        # Load Pretrained ResNet34
        self.base = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Modify first layer for Grayscale (1 channel)
        # Sum weights across the channel dimension to preserve intensity patterns
        w = self.base.conv1.weight
        self.base.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.base.conv1.weight = nn.Parameter(w.sum(dim=1, keepdim=True))

        # Encoder Blocks
        self.encoder0 = nn.Sequential(
            self.base.conv1, self.base.bn1, self.base.relu
        )  # 64, H/2
        self.encoder1 = nn.Sequential(self.base.maxpool, self.base.layer1)  # 64, H/4
        self.encoder2 = self.base.layer2  # 128, H/8
        self.encoder3 = self.base.layer3  # 256, H/16
        self.encoder4 = self.base.layer4  # 512, H/32

        # Depth Injection Projector
        self.depth_projector = nn.Sequential(nn.Linear(1, 32), nn.ReLU(inplace=True))

        # Decoder Blocks
        # Dec4: 512 + 32 (depth) -> 256 (Skip e3: 256)
        self.decoder4 = DecoderBlock(512 + 32, 256)
        # Dec3: 256 -> 128 (Skip e2: 128)
        self.decoder3 = DecoderBlock(256, 128)
        # Dec2: 128 -> 64 (Skip e1: 64)
        self.decoder2 = DecoderBlock(128, 64)
        # Dec1: 64 -> 64 (Skip e0: 64)
        self.decoder1 = DecoderBlock(64, 64)

        # Final Upsampling to Original Resolution (64 -> 128)
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(
                64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x, depth):
        # Encoder
        e0 = self.encoder0(x)  # 64, 64x64
        e1 = self.encoder1(e0)  # 64, 32x32
        e2 = self.encoder2(e1)  # 128, 16x16
        e3 = self.encoder3(e2)  # 256, 8x8
        e4 = self.encoder4(e3)  # 512, 4x4

        # Depth Injection
        # depth: (B, 1) -> (B, 32)
        d_emb = self.depth_projector(depth)
        # Expand to (B, 32, H, W) where H=W=4
        d_emb = d_emb.unsqueeze(2).unsqueeze(3).expand(-1, -1, e4.size(2), e4.size(3))

        # Concatenate at bottleneck
        e4_cat = torch.cat([e4, d_emb], dim=1)

        # Decoder
        d4 = self.decoder4(e4_cat, e3)  # 256, 8x8
        d3 = self.decoder3(d4, e2)  # 128, 16x16
        d2 = self.decoder2(d3, e1)  # 64, 32x32
        d1 = self.decoder1(d2, e0)  # 64, 64x64

        # Final Segmentation
        seg_logits = self.final_up(d1)  # 1, 128x128

        return seg_logits


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, masks, depths in loader:
        images = images.to(device)
        masks = masks.to(device)
        depths = depths.to(device)

        optimizer.zero_grad()

        seg_logits = model(images, depths)

        loss, _ = criterion(seg_logits, masks)

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
        for images, masks, depths in loader:
            images = images.to(device)
            masks = masks.to(device)
            depths = depths.to(device)

            seg_logits = model(images, depths)
            loss, _ = criterion(seg_logits, masks)

            running_loss += loss.item() * images.size(0)

            # Store for metric calculation
            # Center crop to original size (101x101) before metric
            # Current size 128x128.
            # Center is at 64. 101/2 = 50.5. Start: 64-50=14. End: 14+101=115.
            # Or simpler: albumentations PadIfNeeded does center padding.
            # So we center crop.

            preds_prob = torch.sigmoid(seg_logits)

            # Crop to 101x101
            h, w = preds_prob.shape[2], preds_prob.shape[3]
            start_h = (h - config.IMG_ORIG_SIZE) // 2
            start_w = (w - config.IMG_ORIG_SIZE) // 2
            end_h = start_h + config.IMG_ORIG_SIZE
            end_w = start_w + config.IMG_ORIG_SIZE

            preds_crop = preds_prob[:, :, start_h:end_h, start_w:end_w]
            masks_crop = masks[:, :, start_h:end_h, start_w:end_w]

            all_preds.append(preds_crop.cpu())
            all_masks.append(masks_crop.cpu())

    all_preds = torch.cat(all_preds, dim=0)
    all_masks = torch.cat(all_masks, dim=0)

    # Calculate mAP
    map_score = utils.do_kaggle_metric(all_preds, all_masks, threshold=0.5)

    return running_loss / len(loader.dataset), map_score, all_preds, all_masks


def find_best_threshold(preds, masks):
    """
    Linear search for the best binarization threshold on the validation set.
    """
    best_thresh = 0.5
    best_score = 0.0

    # Search range 0.2 to 0.8
    for t in np.arange(0.2, 0.85, 0.05):
        score = utils.do_kaggle_metric(preds, masks, threshold=t)
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def run_training():
    utils.set_seed(config.SEED)
    device = config.DEVICE

    # Data Loaders
    train_loader, val_loader, depth_mean, depth_std = dataset.get_train_val_loaders(
        load_cached_data=True
    )

    # Model Setup
    model = WideLinkNet34().to(device)
    criterion = losses.SaltNetLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-2
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.EPOCHS, eta_min=1e-6
    )

    best_map = 0.0
    patience = 10
    patience_counter = 0

    print(f"Starting training for {config.EPOCHS} epochs on {device}...")

    for epoch in range(config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_map, _, _ = validate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch+1}/{config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val mAP: {val_map:.10f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save Best Model
        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), config.CHECKPOINT_PATH)
            print(f"  >>> Model Saved! New Best mAP: {best_map:.10f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    return depth_mean, depth_std


def predict_and_submit(depth_mean, depth_std):
    device = config.DEVICE

    # Load Best Model
    model = WideLinkNet34().to(device)
    if os.path.exists(config.CHECKPOINT_PATH):
        model.load_state_dict(torch.load(config.CHECKPOINT_PATH, map_location=device))
        print("Loaded best model for inference.")
    else:
        print("Warning: No checkpoint found. Using untrained model.")

    model.eval()

    # Optimize Threshold on Validation Set
    _, val_loader, _, _ = dataset.get_train_val_loaders(load_cached_data=True)
    criterion = losses.SaltNetLoss()  # Dummy for validate function
    _, _, val_preds, val_masks = validate(model, val_loader, criterion, device)

    best_thresh, best_val_score = find_best_threshold(val_preds, val_masks)
    print(f"Optimized Threshold: {best_thresh:.4f} (Val mAP: {best_val_score:.10f})")

    # Test Inference
    test_loader = dataset.get_test_loader(depth_mean, depth_std, load_cached_data=True)

    submission_data = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for images, ids, depths in test_loader:
            images = images.to(device)
            depths = depths.to(device)

            # TTA: Original
            logits = model(images, depths)
            probs = torch.sigmoid(logits)

            # TTA: Flip
            images_flip = torch.flip(images, dims=[3])
            logits_flip = model(images_flip, depths)
            probs_flip = torch.sigmoid(logits_flip)
            probs_flip = torch.flip(probs_flip, dims=[3])

            # Average
            avg_probs = (probs + probs_flip) / 2.0

            # Crop to 101x101
            h, w = avg_probs.shape[2], avg_probs.shape[3]
            start_h = (h - config.IMG_ORIG_SIZE) // 2
            start_w = (w - config.IMG_ORIG_SIZE) // 2
            end_h = start_h + config.IMG_ORIG_SIZE
            end_w = start_w + config.IMG_ORIG_SIZE

            avg_probs = avg_probs[:, :, start_h:end_h, start_w:end_w]

            # Binarize and Encode
            preds_bin = (avg_probs > best_thresh).cpu().numpy().astype(np.uint8)

            for i in range(len(ids)):
                mask = preds_bin[i, 0]
                rle = utils.rle_encode(mask)
                submission_data.append([ids[i], rle])

    # Save Submission
    df_sub = pd.DataFrame(submission_data, columns=["id", "rle_mask"])
    df_sub.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")


def main():
    # Run Training
    d_mean, d_std = run_training()

    # Run Inference
    predict_and_submit(d_mean, d_std)


# Execute the pipeline
if __name__ == "__main__":
    main()

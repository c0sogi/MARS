import os
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import timm
from torchvision.ops import DeformConv2d
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import KuzushijiDataset


# --- Helper Functions ---


def gather_feat(feat, ind, mask=None):
    """
    Gathers values from a feature map at specific indices.
    Used to extract predictions at ground truth center locations.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = feat.gather(
        1, ind.unsqueeze(2).expand(ind.size(0), ind.size(1), feat.size(2))
    )
    return feat


# --- Model Architecture ---


class DCNHead(nn.Module):
    def __init__(self, in_channels, out_channels, head_conv=64):
        super(DCNHead, self).__init__()
        # Offset convolution: predicts 2 * kernel_size * kernel_size offsets
        # 3x3 kernel -> 9 locations * 2 (x,y) = 18 channels
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)

        # Deformable Convolution
        self.dcn = DeformConv2d(in_channels, head_conv, kernel_size=3, padding=1)

        self.bn = nn.BatchNorm2d(head_conv)
        self.relu = nn.ReLU(inplace=True)

        # Final projection
        self.final = nn.Conv2d(head_conv, out_channels, kernel_size=1)

        # Weight initialization
        nn.init.kaiming_normal_(self.dcn.weight, mode="fan_out", nonlinearity="relu")
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)
        nn.init.normal_(self.final.weight, std=0.001)
        nn.init.constant_(self.final.bias, 0)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.dcn(x, offset)
        x = self.bn(x)
        x = self.relu(x)
        x = self.final(x)
        return x


class HRNetCenterNet(nn.Module):
    def __init__(self):
        super(HRNetCenterNet, self).__init__()
        self.num_classes = Config.NUM_CLASSES

        # Load HRNet-W32 backbone
        # features_only=True returns a list of feature maps from different stages
        self.backbone = timm.create_model(
            "hrnet_w32", pretrained=True, features_only=True
        )

        # Determine input channels for the heads
        # HRNet-W32 high-res stream (index 0) typically has 32 channels
        dummy_input = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            features = self.backbone(dummy_input)
        in_channels = features[1].shape[1]

        # Define Heads
        self.hm_head = DCNHead(in_channels, self.num_classes)
        self.wh_head = DCNHead(in_channels, 2)
        self.reg_head = DCNHead(in_channels, 2)

        # Initialize heatmap bias to -2.19 (standard for Focal Loss)
        self.hm_head.final.bias.data.fill_(-2.19)

    def forward(self, x):
        features = self.backbone(x)
        # Use the highest resolution feature map (stride 4)
        x = features[1]

        hm = self.hm_head(x)
        wh = self.wh_head(x)
        reg = self.reg_head(x)

        # Apply sigmoid to heatmap to get probabilities
        hm = torch.sigmoid(hm)

        return hm, wh, reg


# --- Loss Function ---


class KuzushijiLoss(nn.Module):
    def __init__(self):
        super(KuzushijiLoss, self).__init__()

    def focal_loss(self, preds, targets):
        """
        Modified Focal Loss (Penalty Reduced)
        """
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()

        neg_weights = torch.pow(1 - targets, 4)

        loss = 0

        # Clamp predictions to avoid log(0)
        preds = torch.clamp(preds, 1e-6, 1 - 1e-6)

        pos_loss = torch.log(preds) * torch.pow(1 - preds, 2) * pos_inds
        neg_loss = torch.log(1 - preds) * torch.pow(preds, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos

        return loss

    def reg_l1_loss(self, preds, targets, mask):
        """
        L1 Loss for regression targets (WH, Offsets)
        """
        # preds: (B, 2, H, W) -> Gathered to (B, K, 2)
        # targets: (B, K, 2)
        # mask: (B, K)

        expand_mask = mask.unsqueeze(2).expand_as(preds).float()
        loss = F.l1_loss(preds * expand_mask, targets * expand_mask, reduction="sum")
        loss = loss / (mask.float().sum() + 1e-4)
        return loss

    def forward(self, outputs, batch):
        hm_pred, wh_pred, reg_pred = outputs

        hm_true = batch["hm"].to(Config.DEVICE)
        wh_true = batch["wh"].to(Config.DEVICE)
        reg_true = batch["reg"].to(Config.DEVICE)
        ind_true = batch["ind"].to(Config.DEVICE)
        reg_mask = batch["reg_mask"].to(Config.DEVICE)

        # Heatmap Loss
        loss_hm = self.focal_loss(hm_pred, hm_true)

        # Regression Losses (Width/Height & Offset)
        # We must gather the predictions at the ground truth center indices
        wh_pred_gathered = transpose_and_gather_feat(wh_pred, ind_true)
        reg_pred_gathered = transpose_and_gather_feat(reg_pred, ind_true)

        loss_wh = self.reg_l1_loss(wh_pred_gathered, wh_true, reg_mask)
        loss_reg = self.reg_l1_loss(reg_pred_gathered, reg_true, reg_mask)

        # Weighted sum (Weights can be tuned, standard: 1.0, 0.1, 1.0)
        total_loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg

        return total_loss, loss_hm, loss_wh, loss_reg


# --- Training & Inference Logic ---


def train_model():
    Config.setup()
    Config.seed_everything(Config.SEED)

    # Data Loaders
    train_dataset = KuzushijiDataset(split="train")
    val_dataset = KuzushijiDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Model Setup
    model = HRNetCenterNet().to(Config.DEVICE)
    criterion = KuzushijiLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
    )

    best_val_loss = float("inf")
    save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        model.train()
        train_loss_meter = 0

        for batch in train_loader:
            img = batch["image"].to(Config.DEVICE)

            optimizer.zero_grad()
            outputs = model(img)
            loss, l_hm, l_wh, l_reg = criterion(outputs, batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.GRAD_CLIP)
            optimizer.step()

            train_loss_meter += loss.item()

        train_loss_avg = train_loss_meter / len(train_loader)

        # Validation
        model.eval()
        val_loss_meter = 0
        with torch.no_grad():
            for batch in val_loader:
                img = batch["image"].to(Config.DEVICE)
                outputs = model(img)
                loss, _, _, _ = criterion(outputs, batch)
                val_loss_meter += loss.item()

        val_loss_avg = val_loss_meter / len(val_loader)

        scheduler.step()

        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | Train Loss: {train_loss_avg:.6f} | Val Loss: {val_loss_avg:.6f}"
        )

        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            torch.save(model.state_dict(), save_path)
            print("  -> New best model saved!")

    print("Training complete.")


def predict_and_submit():
    Config.setup()
    Config.seed_everything(Config.SEED)

    # Load Data
    test_dataset = KuzushijiDataset(split="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Load Model
    model = HRNetCenterNet().to(Config.DEVICE)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if not os.path.exists(model_path):
        print("Model weights not found. Please train first.")
        return

    model.load_state_dict(torch.load(model_path, map_location=Config.DEVICE))
    model.eval()

    _, id2char = Config.get_class_mappings()

    results = []

    print("Starting inference...")

    with torch.no_grad():
        for batch in test_loader:
            imgs = batch["image"].to(Config.DEVICE)
            img_ids = batch["image_id"]

            hm, wh, reg = model(imgs)

            # Post-processing
            # 1. MaxPool NMS
            pad = (3 - 1) // 2
            hmax = F.max_pool2d(hm, (3, 3), stride=1, padding=pad)
            keep = (hmax == hm).float()
            hm = hm * keep

            # 2. Top K
            B, C, H, W = hm.shape
            hm = hm.view(B, -1)
            scores, inds = torch.topk(hm, Config.MAX_PREDS)

            clses = inds // (H * W)
            inds = inds % (H * W)

            ys = inds // W
            xs = inds % W

            # 3. Gather Regression Targets
            # reg: (B, 2, H, W) -> (B, K, 2)
            reg = transpose_and_gather_feat(reg, inds)
            reg = reg.cpu()

            xs = xs.cpu().float()
            ys = ys.cpu().float()

            # Add offsets
            xs = xs + reg[..., 0]
            ys = ys + reg[..., 1]

            # Scale back to input size (x4)
            xs = xs * 4.0
            ys = ys * 4.0

            scores = scores.cpu().numpy()
            clses = clses.cpu().numpy()
            xs = xs.numpy()
            ys = ys.numpy()

            # 4. Format for submission
            for b in range(B):
                img_id = img_ids[b]
                label_strs = []

                # Filter by confidence
                valid_mask = scores[b] > Config.CONF_THRESHOLD

                valid_scores = scores[b][valid_mask]
                valid_clses = clses[b][valid_mask]
                valid_xs = xs[b][valid_mask]
                valid_ys = ys[b][valid_mask]

                # Transform coordinates back to original image space
                # Note: The dataset resizing maintained aspect ratio with padding.
                # However, the metric allows points within the box.
                # Since we predicted on 1024x1024 padded image, we need to map back.
                # But wait, the test set images are various sizes.
                # The `KuzushijiDataset` uses `get_affine_transform`.
                # We need to invert that transform.
                # Since we don't have the original image size in the batch easily unless we modify dataset,
                # we can rely on the fact that the affine transform was just scaling + padding.
                # Actually, `KuzushijiDataset` does not return original size or matrix.
                # We need to re-read the image or store metadata.
                # To keep it simple and robust within this file:
                # We will reload the image size using cv2 here or assume the dataset can provide it.
                # The dataset `__getitem__` doesn't return metadata.
                # We can read the image file again using `img_id`.

                # Construct path (assuming test set structure)
                # Test images are in input/test_images/
                path = os.path.join(Config.INPUT_DIR, "test_images", f"{img_id}.jpg")
                if not os.path.exists(path):
                    # Fallback if extension differs or path issue
                    path = os.path.join(
                        Config.INPUT_DIR, "test_images", f"{img_id}.jpg"
                    )

                orig_img = cv2.imread(path)
                if orig_img is None:
                    oh, ow = 1024, 1024  # Fallback
                else:
                    oh, ow = orig_img.shape[:2]

                # Re-calculate scale used in dataset
                scale = min(Config.INPUT_SIZE / ow, Config.INPUT_SIZE / oh)

                # Padding (translation)
                nw = int(ow * scale)
                nh = int(oh * scale)
                tx = (Config.INPUT_SIZE - nw) / 2
                ty = (Config.INPUT_SIZE - nh) / 2

                # Inverse transform
                # x_orig = (x_pred - tx) / scale
                # y_orig = (y_pred - ty) / scale

                final_xs = (valid_xs - tx) / scale
                final_ys = (valid_ys - ty) / scale

                for i in range(len(valid_scores)):
                    cid = valid_clses[i]
                    x_p = int(final_xs[i])
                    y_p = int(final_ys[i])

                    # Bounds check
                    x_p = max(0, min(x_p, ow - 1))
                    y_p = max(0, min(y_p, oh - 1))

                    unicode_char = id2char[cid]

                    # Submission format: "Unicode X Y"
                    label_strs.append(f"{unicode_char} {x_p} {y_p}")

                prediction_string = " ".join(label_strs)
                results.append({"image_id": img_id, "labels": prediction_string})

    # Save Submission
    sub_df = pd.DataFrame(results)
    sub_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

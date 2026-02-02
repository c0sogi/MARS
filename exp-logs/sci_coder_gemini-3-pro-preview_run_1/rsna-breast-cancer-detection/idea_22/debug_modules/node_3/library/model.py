import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops
import timm
import pandas as pd
import numpy as np
from tqdm import tqdm
from library.config import Config
from library.utils import probabilistic_f1, set_seed
from library.data import get_dataloaders

# -------------------------------------------------------------------------
# Layers & Blocks
# -------------------------------------------------------------------------


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block for Attention Gating.
    """

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class DeformableAlignmentModule(nn.Module):
    """
    Pyramid Deformable Alignment Module.
    Predicts offsets from concatenated features and aligns the contralateral feature map.
    """

    def __init__(self, in_channels):
        super(DeformableAlignmentModule, self).__init__()
        self.kernel_size = 3
        self.padding = 1

        # Predicts offsets (2*k^2) and mask (k^2) -> Total 3*k^2 channels
        # Input is concatenation of Target and Contra features (2 * in_channels)
        self.offset_conv = nn.Conv2d(
            in_channels * 2,
            3 * self.kernel_size * self.kernel_size,
            kernel_size=3,
            padding=1,
        )

        # Deformable Convolution Weights
        # We learn a convolution on the resampled features to best match the target
        self.dcn_weight = nn.Parameter(
            torch.empty(in_channels, in_channels, self.kernel_size, self.kernel_size)
        )
        self.dcn_bias = nn.Parameter(torch.empty(in_channels))

        # Initialization
        nn.init.kaiming_uniform_(self.dcn_weight, nonlinearity="relu")
        nn.init.constant_(self.dcn_bias, 0)
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

    def forward(self, x_target, x_contra):
        # 1. Predict Offsets and Mask
        combined = torch.cat([x_target, x_contra], dim=1)
        out = self.offset_conv(combined)

        k2 = self.kernel_size * self.kernel_size
        offset, mask = torch.split(out, [2 * k2, k2], dim=1)
        mask = torch.sigmoid(mask)

        # 2. Apply Deformable Convolution
        aligned_contra = torchvision.ops.deform_conv2d(
            input=x_contra,
            offset=offset,
            weight=self.dcn_weight,
            bias=self.dcn_bias,
            padding=self.padding,
            mask=mask,
        )

        return aligned_contra


class SiameseEfficientNet(nn.Module):
    """
    Deformable Attention-Gated Siamese EfficientNet-B2.
    """

    def __init__(self):
        super(SiameseEfficientNet, self).__init__()

        # Backbone: EfficientNet-B2
        # in_chans=3 (Image, Age, Implant)
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=Config.PRETRAINED,
            features_only=True,
            in_chans=Config.NUM_CHANNELS,
            out_indices=(2, 3, 4),  # P3, P4, P5
        )

        # Determine feature channels dynamically
        dummy_input = torch.randn(1, Config.NUM_CHANNELS, 256, 256)
        features = self.backbone(dummy_input)
        self.feature_channels = [f.shape[1] for f in features]

        # Alignment, Difference, and Attention Modules for each level
        self.align_modules = nn.ModuleList()
        self.se_blocks = nn.ModuleList()

        for ch in self.feature_channels:
            self.align_modules.append(DeformableAlignmentModule(ch))
            self.se_blocks.append(SEBlock(ch))

        # Fusion Head
        # Concatenate: GlobalAvgPool(Target) + GlobalAvgPool(Diff) for each level
        # Total dims = Sum(ch) * 2
        total_dim = sum(self.feature_channels) * 2

        self.head = nn.Sequential(
            nn.Dropout(p=Config.DROP_RATE), nn.Linear(total_dim, 1)
        )

    def forward_features(self, x):
        return self.backbone(x)

    def forward(self, x_target, x_contra):
        # Extract features (Shared Backbone)
        feats_target = self.forward_features(x_target)
        feats_contra = self.forward_features(x_contra)

        global_descriptors = []

        for i, (ft, fc) in enumerate(zip(feats_target, feats_contra)):
            # 1. Deformable Alignment
            fc_aligned = self.align_modules[i](ft, fc)

            # 2. Difference
            diff = ft - fc_aligned

            # 3. Attention Gating
            diff_attended = self.se_blocks[i](diff)

            # 4. Pooling
            # Target features (Context)
            pool_t = F.adaptive_avg_pool2d(ft, 1).flatten(1)
            # Difference features (Asymmetry Signal)
            pool_d = F.adaptive_avg_pool2d(diff_attended, 1).flatten(1)

            global_descriptors.append(pool_t)
            global_descriptors.append(pool_d)

        # Concatenate all descriptors
        embedding = torch.cat(global_descriptors, dim=1)

        # Classification
        logits = self.head(embedding)
        return logits


# -------------------------------------------------------------------------
# Training & Inference Logic
# -------------------------------------------------------------------------


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch in tqdm(loader, desc="Training", leave=False):
        target_img = batch["target"].to(device)
        contra_img = batch["contra"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()

        logits = model(target_img, contra_img)
        loss = criterion(logits, labels)

        loss.backward()

        # Disable Gradient Clipping as per strategy
        if Config.MAX_GRAD_NORM is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.MAX_GRAD_NORM)

        optimizer.step()

        running_loss += loss.item() * target_img.size(0)

        probs = torch.sigmoid(logits).detach().cpu()
        all_preds.extend(probs.numpy())
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

    return epoch_loss, pf1


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation", leave=False):
            target_img = batch["target"].to(device)
            contra_img = batch["contra"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            logits = model(target_img, contra_img)
            loss = criterion(logits, labels)

            running_loss += loss.item() * target_img.size(0)

            probs = torch.sigmoid(logits).cpu()
            all_preds.extend(probs.numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    pf1 = probabilistic_f1(np.array(all_targets), np.array(all_preds))

    return epoch_loss, pf1


def fit():
    """
    Main training loop.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Data
    train_loader, val_loader, _ = get_dataloaders()

    # Model
    model = SiameseEfficientNet().to(device)

    # Loss (Weighted BCE)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.NUM_EPOCHS, eta_min=Config.ETA_MIN
    )

    # Tracking
    best_pf1 = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print(f"Starting training on {device} for {Config.NUM_EPOCHS} epochs...")

    for epoch in range(Config.NUM_EPOCHS):
        train_loss, train_pf1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS}")
        print(f"  Train Loss: {train_loss:.6f} | Train pF1: {train_pf1:.6f}")
        print(f"  Val Loss:   {val_loss:.6f} | Val pF1:   {val_pf1:.6f}")

        # Checkpointing & Early Stopping
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print("  [Saved Best Model]")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break

    print(f"Training complete. Best Val pF1: {best_pf1:.6f}")
    return best_model_path


def generate_submission():
    """
    Generates predictions for the test set and saves submission.csv.
    """
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Data
    _, _, test_loader = get_dataloaders()

    # Load Model
    model = SiameseEfficientNet().to(device)
    model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(
            "Warning: No trained model found. Using random initialization (for debugging only)."
        )

    model.eval()

    results = []  # List of (prediction_id, probability)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Inference"):
            target_img = batch["target"].to(device)
            contra_img = batch["contra"].to(device)
            pred_ids = batch["prediction_id"]

            logits = model(target_img, contra_img)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()

            for pid, prob in zip(pred_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Aggregate predictions (Max over views)
    df_res = pd.DataFrame(results)

    # Group by prediction_id and take max
    df_sub = df_res.groupby("prediction_id", as_index=False)["cancer"].max()

    # Ensure all prediction_ids from sample submission are present
    # (Though test_loader covers all test.csv rows, so we should be good)

    # Save
    df_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

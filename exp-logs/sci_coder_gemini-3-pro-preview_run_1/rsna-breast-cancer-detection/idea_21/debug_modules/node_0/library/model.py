import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from library.config import Config
from library.utils import set_seed, setup_logger, probabilistic_f1, load_image

# ==========================================
# Model Architecture
# ==========================================


class SpatialAlignmentModule(nn.Module):
    """
    Predicts a deformation field (flow) to align the contralateral feature map
    to the target feature map.
    """

    def __init__(self, in_channels):
        super().__init__()
        # Predict deformation field from concatenated features (2 * in_channels)
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels, in_channels // 2, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels // 2, 2, kernel_size=3, padding=1, bias=False
            ),  # Output: 2 channels for (dx, dy)
        )
        # Initialize flow to 0 (identity transform)
        self.conv[-1].weight.data.zero_()

    def forward(self, target_feat, contra_feat):
        # target_feat, contra_feat: [B, C, H, W]
        x = torch.cat([target_feat, contra_feat], dim=1)
        flow = self.conv(x)  # [B, 2, H, W]
        return flow

    def warp(self, x, flow):
        # x: [B, C, H, W], flow: [B, 2, H_flow, W_flow]
        B, C, H, W = x.size()

        # Upsample flow to x size if necessary
        if flow.shape[2:] != x.shape[2:]:
            flow = F.interpolate(flow, size=(H, W), mode="bilinear", align_corners=True)

        # Create base grid range [-1, 1]
        xx = torch.linspace(-1, 1, W, device=x.device)
        yy = torch.linspace(-1, 1, H, device=x.device)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
        base_grid = (
            torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1)
        )  # [B, H, W, 2]

        # Apply flow
        # Permute flow to [B, H, W, 2] to match grid format
        flow_perm = flow.permute(0, 2, 3, 1)
        final_grid = base_grid + flow_perm

        # Grid sample
        warped = F.grid_sample(
            x, final_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return warped


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""

    def __init__(self, channels, reduction=16):
        super().__init__()
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


class AttentiveDifferenceModule(nn.Module):
    """Computes difference and applies channel attention"""

    def __init__(self, channels):
        super().__init__()
        self.se = SEBlock(channels)

    def forward(self, target_feat, warped_contra_feat):
        diff = target_feat - warped_contra_feat
        diff = self.se(diff)
        return diff


class SiameseEfficientNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b2", pretrained=True):
        super().__init__()

        # Backbone
        # efficientnet_b2 features: P3 (stride 8), P4 (stride 16), P5 (stride 32)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),  # P3, P4, P5
            in_chans=Config.IN_CHANNELS,
        )

        feature_info = self.backbone.feature_info
        channels = [x["num_chs"] for x in feature_info]

        # Alignment Module (Uses P5 features for flow estimation)
        self.alignment = SpatialAlignmentModule(channels[2])

        # Difference Modules for each scale
        self.diff_p3 = AttentiveDifferenceModule(channels[0])
        self.diff_p4 = AttentiveDifferenceModule(channels[1])
        self.diff_p5 = AttentiveDifferenceModule(channels[2])

        # Classification Head
        # Input: Concat(GAP(Target_P3), GAP(Diff_P3), ..., GAP(Target_P5), GAP(Diff_P5))
        total_features = sum(channels) * 2
        self.classifier = nn.Linear(total_features, 1)

    def forward_features(self, x):
        return self.backbone(x)  # Returns [P3, P4, P5]

    def forward(self, x_target, x_contra):
        # Extract features
        feats_target = self.forward_features(x_target)  # [P3, P4, P5]
        feats_contra = self.forward_features(x_contra)  # [P3, P4, P5]

        # 1. Alignment (Estimate flow at P5)
        flow = self.alignment(feats_target[2], feats_contra[2])

        # 2. Warp Contralateral Features at all levels using the flow
        warped_contra_p3 = self.alignment.warp(feats_contra[0], flow)
        warped_contra_p4 = self.alignment.warp(feats_contra[1], flow)
        warped_contra_p5 = self.alignment.warp(feats_contra[2], flow)

        # 3. Compute Attentive Differences
        diff_p3 = self.diff_p3(feats_target[0], warped_contra_p3)
        diff_p4 = self.diff_p4(feats_target[1], warped_contra_p4)
        diff_p5 = self.diff_p5(feats_target[2], warped_contra_p5)

        # 4. Pooling
        # Target features
        gap_tgt_p3 = F.adaptive_avg_pool2d(feats_target[0], 1).flatten(1)
        gap_tgt_p4 = F.adaptive_avg_pool2d(feats_target[1], 1).flatten(1)
        gap_tgt_p5 = F.adaptive_avg_pool2d(feats_target[2], 1).flatten(1)

        # Difference features
        gap_diff_p3 = F.adaptive_avg_pool2d(diff_p3, 1).flatten(1)
        gap_diff_p4 = F.adaptive_avg_pool2d(diff_p4, 1).flatten(1)
        gap_diff_p5 = F.adaptive_avg_pool2d(diff_p5, 1).flatten(1)

        # 5. Concatenate
        features = torch.cat(
            [gap_tgt_p3, gap_diff_p3, gap_tgt_p4, gap_diff_p4, gap_tgt_p5, gap_diff_p5],
            dim=1,
        )

        # 6. Classify
        logits = self.classifier(features)
        return logits


# ==========================================
# Data Loading
# ==========================================


class SiameseDataset(Dataset):
    def __init__(self, df, transform=None, mode="train"):
        self.df = df
        self.transform = transform
        self.mode = mode

        # Group by patient and view to speed up pair finding
        # Key: (patient_id, view) -> DataFrame subset
        self.grouped = self.df.groupby(["patient_id", "view"])

        # Pre-compute pairs list
        self.pairs = []
        for idx, row in self.df.iterrows():
            patient_id = row["patient_id"]
            view = row["view"]
            laterality = row["laterality"]

            # Target info
            target_path = os.path.join(Config.INPUT_DIR, row["file_path"])

            # Find contralateral: Same patient, same view, opposite laterality
            opp_laterality = "R" if laterality == "L" else "L"

            try:
                # Get all images for this patient+view
                group = self.grouped.get_group((patient_id, view))
                # Filter for opposite laterality
                contra_rows = group[group["laterality"] == opp_laterality]
            except KeyError:
                contra_rows = pd.DataFrame()

            if not contra_rows.empty:
                # Use the first match
                c_path = os.path.join(
                    Config.INPUT_DIR, contra_rows.iloc[0]["file_path"]
                )
            else:
                c_path = None

            self.pairs.append(
                {
                    "target_path": target_path,
                    "contra_path": c_path,
                    "age": row["age"],
                    "implant": row["implant"],
                    "label": row["cancer"] if "cancer" in row else 0,
                    "prediction_id": (
                        row["prediction_id"]
                        if "prediction_id" in row
                        else f"{patient_id}_{laterality}"
                    ),
                }
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        data = self.pairs[idx]

        # Load Target Image
        try:
            img_target = load_image(data["target_path"], size=Config.IMG_SIZE)
        except Exception as e:
            # Fail loudly as per requirements
            raise ValueError(f"Error loading target {data['target_path']}: {e}")

        # Load Contralateral Image
        if data["contra_path"] and os.path.exists(data["contra_path"]):
            try:
                img_contra = load_image(data["contra_path"], size=Config.IMG_SIZE)
            except:
                # If corrupt, treat as missing
                img_contra = np.zeros_like(img_target)
        else:
            # Missing contralateral
            img_contra = np.zeros_like(img_target)

        # Augmentation
        # Apply synchronized geometric transforms
        if self.transform:
            res = self.transform(image=img_target, image_contra=img_contra)
            img_target = res["image"]
            img_contra = res["image_contra"]
        else:
            # Fallback (should not happen in this pipeline)
            img_target = torch.from_numpy(img_target).float().unsqueeze(0)
            img_contra = torch.from_numpy(img_contra).float().unsqueeze(0)

        # Metadata Channels
        # Age: Standard Scaling (Mean ~58.7, Std ~10.0)
        age_val = (data["age"] - 58.7) / 10.0
        if np.isnan(age_val):
            age_val = 0.0

        implant_val = 1.0 if data["implant"] == 1 else 0.0

        # Create Metadata Maps (spatially broadcasted)
        # img tensor is [C, H, W]
        _, h, w = img_target.shape

        age_map = torch.full((1, h, w), age_val, dtype=torch.float32)
        implant_map = torch.full((1, h, w), implant_val, dtype=torch.float32)

        # Concatenate: Image (1) + Age (1) + Implant (1) = 3 Channels
        input_target = torch.cat([img_target, age_map, implant_map], dim=0)
        input_contra = torch.cat([img_contra, age_map, implant_map], dim=0)

        return (
            input_target,
            input_contra,
            torch.tensor(data["label"], dtype=torch.float32),
            data["prediction_id"],
        )


def get_transforms(mode="train"):
    if mode == "train":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(limit=15, p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.5
                ),
                A.Normalize(mean=(0.2,), std=(0.22,)),
                ToTensorV2(),
            ],
            additional_targets={"image_contra": "image"},
        )
    else:
        return A.Compose(
            [A.Normalize(mean=(0.2,), std=(0.22,)), ToTensorV2()],
            additional_targets={"image_contra": "image"},
        )


# ==========================================
# Training & Evaluation
# ==========================================


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    running_loss = 0.0

    for i, (tgt, contra, label, _) in enumerate(loader):
        tgt, contra, label = tgt.to(device), contra.to(device), label.to(device)

        optimizer.zero_grad()

        logits = model(tgt, contra).squeeze(1)
        loss = criterion(logits, label)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    labels = []

    with torch.no_grad():
        for tgt, contra, label, _ in loader:
            tgt, contra, label = tgt.to(device), contra.to(device), label.to(device)

            logits = model(tgt, contra).squeeze(1)
            loss = criterion(logits, label)

            probs = torch.sigmoid(logits).cpu().numpy()

            running_loss += loss.item()
            preds.extend(probs)
            labels.extend(label.cpu().numpy())

    pf1 = probabilistic_f1(labels, preds)
    return running_loss / len(loader), pf1


# ==========================================
# Main Execution Functions
# ==========================================


def prepare_data(debug=False, load_cached_data=True):
    """
    Loads metadata and prepares DataLoaders.
    """
    # Load raw metadata
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)

    if debug:
        df_train = df_train.sample(
            n=Config.DEBUG_SAMPLE_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SAMPLE_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    # Datasets
    train_dataset = SiameseDataset(
        df_train, transform=get_transforms("train"), mode="train"
    )
    val_dataset = SiameseDataset(df_val, transform=get_transforms("val"), mode="val")

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    return train_loader, val_loader


def run_training(debug=False, epochs=None):
    # Setup
    Config.setup(debug=debug, epochs=epochs)
    set_seed(Config.SEED)
    logger = setup_logger(os.path.join(Config.WORKING_DIR, "training.log"))

    device = torch.device(Config.DEVICE)

    # Data
    logger.info("Preparing data...")
    train_loader, val_loader = prepare_data(debug=debug)

    # Model
    logger.info(f"Initializing model: {Config.BACKBONE}")
    model = SiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=True)
    model = model.to(device)

    # Optimization
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([Config.POS_WEIGHT]).to(device)
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    # Training Loop
    best_pf1 = 0.0

    logger.info("Starting training...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        val_loss, val_pf1 = validate(model, val_loader, criterion, device)

        scheduler.step()

        logger.info(
            f"Epoch {epoch+1}/{Config.EPOCHS} - "
            f"Train Loss: {train_loss:.6f} - "
            f"Val Loss: {val_loss:.6f} - "
            f"Val pF1: {val_pf1:.10f}"
        )

        # Save Best
        if val_pf1 > best_pf1:
            best_pf1 = val_pf1
            torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            logger.info(f"New best model saved with pF1: {best_pf1:.10f}")

    logger.info("Training complete.")


def generate_submission(debug=False):
    # Setup
    Config.setup(debug=debug)
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)

    # Load Model
    model = SiameseEfficientNet(backbone_name=Config.BACKBONE, pretrained=False)

    if os.path.exists(Config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))
        print(f"Loaded model from {Config.MODEL_SAVE_PATH}")
    else:
        print(
            "Warning: No trained model found. Using random weights (for debugging only)."
        )

    model = model.to(device)
    model.eval()

    # Load Test Data
    df_test = pd.read_csv(Config.TEST_CSV)
    if debug:
        df_test = df_test.head(100)

    test_dataset = SiameseDataset(
        df_test, transform=get_transforms("test"), mode="test"
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # Predict
    results = []
    with torch.no_grad():
        for tgt, contra, _, pred_ids in test_loader:
            tgt, contra = tgt.to(device), contra.to(device)
            logits = model(tgt, contra).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()

            for pid, prob in zip(pred_ids, probs):
                results.append({"prediction_id": pid, "cancer": prob})

    # Aggregate (Max per prediction_id)
    df_res = pd.DataFrame(results)
    df_submission = df_res.groupby("prediction_id")["cancer"].max().reset_index()

    # Save
    out_path = os.path.join(Config.WORKING_DIR, Config.SUBMISSION_PATH)
    df_submission.to_csv(out_path, index=False)
    print(f"Submission saved to {out_path}")

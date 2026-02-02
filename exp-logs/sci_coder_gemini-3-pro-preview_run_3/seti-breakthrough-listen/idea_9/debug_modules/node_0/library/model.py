import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import timm
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

from library.config import Config
from library.dataset import SETIDataset, get_transforms
from library.utils import (
    seed_everything,
    mixup_data,
    mixup_criterion,
    get_score,
    AverageMeter,
)

# ==========================================
# Model Architecture
# ==========================================


class GeM(nn.Module):
    """
    Generalized Mean Pooling layer.
    """

    def __init__(self, p=3.0, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        # Global pooling: output (B, C)
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3.0, eps=1e-6):
        return (
            F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1)))
            .pow(1.0 / p)
            .squeeze(-1)
            .squeeze(-1)
        )

    def __repr__(self):
        return (
            self.__class__.__name__
            + "("
            + "p="
            + "{:.4f}".format(self.p.data.tolist()[0])
            + ", "
            + "eps="
            + str(self.eps)
            + ")"
        )


class SiameseEfficientNetV2(nn.Module):
    """
    Siamese EfficientNet-V2 with Adaptive Difference and GeM Pooling.
    """

    def __init__(
        self,
        backbone_name=Config.BACKBONE,
        pretrained=True,
        in_channels=Config.IN_CHANNELS,
    ):
        super(SiameseEfficientNetV2, self).__init__()

        # Load Backbone
        # We use num_classes=0 to get the feature map before the classifier
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="",
            in_chans=in_channels,
            features_only=True,
            out_indices=(4,),  # Get the last feature map
        )

        # Get feature dimension dynamically
        dummy_input = torch.randn(1, in_channels, Config.IMG_HEIGHT, Config.IMG_WIDTH)
        with torch.no_grad():
            features = self.backbone(dummy_input)
            # features is a list because features_only=True
            last_feat = features[-1]
            self.num_features = last_feat.shape[1]

        # Adaptive Difference Scale Parameter
        # Shape: (1, C, 1, 1) for channel-wise scaling
        self.scale = nn.Parameter(torch.ones(1, self.num_features, 1, 1))

        # GeM Pooling Layers
        self.gem_on = GeM(p=Config.GEM_P)
        self.gem_off = GeM(p=Config.GEM_P)
        self.gem_diff = GeM(p=Config.GEM_P)

        # Classifier Head
        # Input dimension is 3 * num_features (On + Off + Diff)
        self.head = nn.Sequential(
            nn.Dropout(Config.DROP_RATE), nn.Linear(self.num_features * 3, 1)
        )

    def forward(self, x):
        # x shape: (B, 6, H, W)
        # Split into On-Target (Stream A) and Off-Target (Stream B)
        # According to dataset.py, indices are [0, 2, 4, 1, 3, 5]
        # So first 3 are On, last 3 are Off.
        x_on = x[:, :3, :, :]
        x_off = x[:, 3:, :, :]

        # Shared Backbone Feature Extraction
        # backbone(x) returns a list of features, we want the last one
        f_on = self.backbone(x_on)[-1]
        f_off = self.backbone(x_off)[-1]

        # Adaptive Difference
        # F_diff = F_on - (w * F_off)
        f_diff = f_on - (self.scale * f_off)

        # GeM Pooling
        v_on = self.gem_on(f_on)
        v_off = self.gem_off(f_off)
        v_diff = self.gem_diff(f_diff)

        # Concatenate
        v_cat = torch.cat([v_on, v_off, v_diff], dim=1)

        # Classification
        logits = self.head(v_cat)

        return logits


# ==========================================
# Training & Inference Pipeline
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    losses = AverageMeter()

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        # Mixup
        images, targets_a, targets_b, lam = mixup_data(
            images, targets, Config.MIXUP_ALPHA, device
        )

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    valid_labels = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, targets)

            losses.update(loss.item(), images.size(0))

            preds.append(torch.sigmoid(outputs).cpu().numpy())
            valid_labels.append(targets.cpu().numpy())

    preds = np.concatenate(preds)
    valid_labels = np.concatenate(valid_labels)
    auc = get_score(valid_labels, preds)

    return losses.avg, auc


def inference(model, loader, device):
    model.eval()
    preds = []

    # Test Time Augmentation (TTA)
    # 1. Original
    # 2. Horizontal Flip (Time Reversal) -> dim -1 (W)
    # 3. Vertical Flip (Freq Inversion) -> dim -2 (H)
    # 4. H + V Flip

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            batch_preds = []

            # TTA Loop
            for flip_h in [False, True]:
                for flip_v in [False, True]:
                    img_aug = images.clone()
                    if flip_h:
                        img_aug = torch.flip(img_aug, dims=[-1])
                    if flip_v:
                        img_aug = torch.flip(img_aug, dims=[-2])

                    out = model(img_aug).squeeze(1)
                    batch_preds.append(torch.sigmoid(out).cpu().numpy())

            # Average predictions across TTA
            batch_preds = np.mean(batch_preds, axis=0)
            preds.append(batch_preds)

    return np.concatenate(preds)


def run_experiment():
    # 1. Setup
    seed_everything(Config.SEED)
    Config.setup()
    device = torch.device(Config.DEVICE)

    print(
        f"Starting experiment: Idea 9 (Siamese EfficientNet-V2 + GeM + Adaptive Diff)"
    )
    print(f"Device: {device}")

    # 2. Data Loading
    print("Loading metadata...")
    df_train = pd.read_csv(Config.TRAIN_CSV)
    df_val = pd.read_csv(Config.VAL_CSV)
    df_test = pd.read_csv(Config.TEST_CSV)

    if Config.DEBUG:
        print(f"Debug mode: sampling {Config.DEBUG_SUBSET_SIZE} rows.")
        df_train = df_train.sample(
            n=Config.DEBUG_SUBSET_SIZE, random_state=Config.SEED
        ).reset_index(drop=True)
        df_val = df_val.sample(
            n=min(len(df_val), Config.DEBUG_SUBSET_SIZE), random_state=Config.SEED
        ).reset_index(drop=True)

    train_dataset = SETIDataset(df_train, transform=get_transforms("train"))
    val_dataset = SETIDataset(df_val, transform=get_transforms("valid"))
    test_dataset = SETIDataset(df_test, transform=get_transforms("test"))

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
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Model Setup
    print("Initializing model...")
    model = SiameseEfficientNetV2(pretrained=Config.PRETRAINED).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR)

    best_auc = 0.0
    best_model_path = os.path.join(Config.WORK_DIR, "best_model.pth")

    # 4. Training Loop
    print(f"Starting training for {Config.EPOCHS} epochs...")
    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{Config.EPOCHS} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Val Loss: {val_loss:.6f} | "
            f"Val AUC: {val_auc:.9f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_auc > best_auc:
            print(f"AUC Improved ({best_auc:.9f} -> {val_auc:.9f}). Saving model.")
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)

    print(f"Training complete. Best Val AUC: {best_auc:.9f}")

    # 5. Inference
    print("Starting inference on test set with TTA...")
    # Load best model
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    predictions = inference(model, test_loader, device)

    # 6. Submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df_test["target"] = predictions
    df_test[["id", "target"]].to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


# Execute the experiment
run_experiment()

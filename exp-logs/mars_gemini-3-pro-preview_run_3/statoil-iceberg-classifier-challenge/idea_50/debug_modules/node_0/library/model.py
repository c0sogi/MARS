import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from library.utils import set_seed, save_checkpoint, AverageMeter
from library.dataset import get_dataloaders

# ==========================================================================================
# Custom Layers
# ==========================================================================================


class MaxMagnitudePool2d(nn.Module):
    """
    Max-Magnitude Pooling.
    Selects the pixel with the largest absolute value within the pooling window,
    but propagates the original signed value.
    """

    def __init__(
        self, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.ceil_mode = ceil_mode

    def forward(self, x):
        # x: (B, C, H, W)

        # 1. Compute absolute values
        x_abs = x.abs()

        # 2. Get indices of the max absolute value
        # return_indices=True returns the linear indices in the flattened spatial dimensions
        _, indices = F.max_pool2d(
            x_abs,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.ceil_mode,
            return_indices=True,
        )

        # 3. Gather original values using the indices
        # Flatten spatial dimensions: (B, C, H*W)
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1)
        indices_flat = indices.view(B, C, -1)

        # Gather selected pixels
        out_flat = torch.gather(x_flat, 2, indices_flat)

        # 4. Reshape back to output spatial dimensions
        # The output shape is determined by the pooling operation logic
        # We can simply use the shape of the indices tensor
        H_out = indices.shape[2]
        W_out = indices.shape[3]

        return out_flat.view(B, C, H_out, W_out)


class LeakySEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block with LeakyReLU in the bottleneck.
    Allows shadow-dominant channels (negative average) to participate in attention.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(channels // reduction, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


# ==========================================================================================
# Model Architecture
# ==========================================================================================


class MPDPCNN(nn.Module):
    """
    Magnitude-Preserving Dual-Polarity CNN.
    """

    def __init__(self):
        super().__init__()

        # --- Backbone (4 Stages) ---
        # Stage 1: 75x75 -> 37x37
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(64)
        self.lrelu1 = nn.LeakyReLU(0.1, inplace=True)
        self.se1 = LeakySEBlock(64, reduction=8)
        self.pool1 = MaxMagnitudePool2d(2, 2)  # 75 -> 37

        # Stage 2: 37x37 -> 18x18
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(128)
        self.lrelu2 = nn.LeakyReLU(0.1, inplace=True)
        self.se2 = LeakySEBlock(128, reduction=8)
        self.pool2 = MaxMagnitudePool2d(2, 2)  # 37 -> 18

        # Stage 3: 18x18 -> 9x9
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(128)
        self.lrelu3 = nn.LeakyReLU(0.1, inplace=True)
        self.se3 = LeakySEBlock(128, reduction=8)
        self.pool3 = MaxMagnitudePool2d(2, 2)  # 18 -> 9

        # Stage 4: 9x9 -> 4x4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True)
        self.bn4 = nn.BatchNorm2d(128)
        self.lrelu4 = nn.LeakyReLU(0.1, inplace=True)
        self.se4 = LeakySEBlock(128, reduction=8)
        self.pool4 = MaxMagnitudePool2d(2, 2)  # 9 -> 4

        # --- Readout (Hierarchical Dual-Polarity) ---
        # Projections to reduce channel dim before flattening
        self.proj3 = nn.Conv2d(128, 64, kernel_size=1)
        self.proj4 = nn.Conv2d(128, 64, kernel_size=1)

        # --- Classification Head ---
        # Input: (64_max + 64_min)_stage3 + (64_max + 64_min)_stage4 + 1_angle = 257
        self.head = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        # --- Initialization ---
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, angle):
        # Stage 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu1(x)
        x = self.se1(x)
        x = self.pool1(x)

        # Stage 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)
        x = self.se2(x)
        x = self.pool2(x)

        # Stage 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)
        x = self.se3(x)
        x = self.pool3(x)
        s3 = x  # 9x9

        # Stage 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)
        x = self.se4(x)
        x = self.pool4(x)
        s4 = x  # 4x4

        # Readout Stage 3
        p3 = self.proj3(s3)  # 128 -> 64
        # Global Max Pooling
        gmax3 = F.adaptive_max_pool2d(p3, 1).view(p3.size(0), -1)
        # Global Min Pooling (via -max(-x))
        gmin3 = -F.adaptive_max_pool2d(-p3, 1).view(p3.size(0), -1)

        # Readout Stage 4
        p4 = self.proj4(s4)  # 128 -> 64
        gmax4 = F.adaptive_max_pool2d(p4, 1).view(p4.size(0), -1)
        gmin4 = -F.adaptive_max_pool2d(-p4, 1).view(p4.size(0), -1)

        # Feature Concatenation
        # 64*4 = 256 features
        features = torch.cat([gmax3, gmin3, gmax4, gmin4], dim=1)

        # Angle Fusion
        angle = angle.view(-1, 1)
        fused = torch.cat([features, angle], dim=1)

        # Classification
        out = self.head(fused)
        return out


# ==========================================================================================
# Training & Evaluation Logic
# ==========================================================================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for batch in loader:
        images = batch["image"].to(device)
        angles = batch["angle"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(images, angles)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))

    return losses.avg


def predict(model, loader, device):
    model.eval()
    ids_all = []
    preds_all = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            angles = batch["angle"].to(device)
            ids = batch["id"]

            outputs = model(images, angles)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()

            ids_all.extend(ids)
            preds_all.extend(probs)

    return ids_all, preds_all


def run_training(
    epochs=75,
    patience=12,
    batch_size=32,
    lr=1e-3,
    seed=42,
    working_dir="./working/idea_50",
):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Data Loading
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        input_dir="./input",
        metadata_dir="./metadata",
        cache_dir=working_dir,
        batch_size=batch_size,
        load_cached_data=True,
        seed=seed,
    )

    # 2. Model Setup
    model = MPDPCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    # Weight decay (L2) is part of AdamW
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # 3. Training Loop
    best_val_loss = float("inf")
    patience_counter = 0
    checkpoint_dir = os.path.join(working_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"Starting training for {epochs} epochs on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Checkpointing & Early Stopping
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_score": best_val_loss,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            checkpoint_dir,
            fold=0,  # Single fold logic for this script
        )

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 4. Inference
    print("Loading best model for inference...")
    best_model_path = os.path.join(checkpoint_dir, "model_best_fold_0.pth")
    load_checkpoint(best_model_path, model)

    print("Generating predictions...")
    ids, probs = predict(model, test_loader, device)

    # 5. Save Submission
    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    df_sub = pd.DataFrame({"id": ids, "is_iceberg": probs})
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


# To run the training:
# run_training()

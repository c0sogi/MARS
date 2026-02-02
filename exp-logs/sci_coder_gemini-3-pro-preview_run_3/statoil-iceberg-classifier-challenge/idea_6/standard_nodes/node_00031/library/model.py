import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from library.utils import get_device, set_seed


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block.
    Recalibrates channel-wise feature responses by explicitly modelling interdependencies between channels.
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
        return x * y.expand_as(x)


class ResidualBlock(nn.Module):
    """
    Residual Block with integrated SE attention.
    Structure: Conv-BN-ReLU-Conv-BN-SE + Shortcut -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # SE Block integrated within the residual unit
        self.se = SEBlock(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE attention to the residual branch features
        out = self.se(out)

        out += residual
        out = self.relu(out)
        return out


class MicroSEResNet(nn.Module):
    """
    Micro-Residual Network with SE Attention.
    Designed for 75x75 3-channel input (HH, HV, Avg).
    Stages: 64 -> 128 -> 128 channels.
    """

    def __init__(self, num_blocks=[2, 2, 2], dropout_rate=0.5):
        super(MicroSEResNet, self).__init__()
        self.in_channels = 64

        # Stem
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Stages
        # Stage 1: 64 channels, 75x75
        self.layer1 = self._make_layer(64, num_blocks[0], stride=1)
        # Stage 2: 128 channels, 38x38 (stride 2)
        self.layer2 = self._make_layer(128, num_blocks[1], stride=2)
        # Stage 3: 128 channels, 19x19 (stride 2)
        self.layer3 = self._make_layer(128, num_blocks[2], stride=2)

        # Head
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Fusion and Classification
        # Input features (128) + Angle (1) = 129
        self.fc1 = nn.Linear(128 + 1, 512)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x, angle):
        # Feature Extraction
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # Global Max Pooling
        out = self.global_max_pool(out)
        out = out.view(out.size(0), -1)  # Flatten -> (B, 128)

        # Fusion with Incidence Angle
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        out = torch.cat([out, angle], dim=1)  # -> (B, 129)

        # Classification Head
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.sigmoid(out)

        return out


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for inputs, angles, labels in dataloader:
        inputs = inputs.to(device)
        angles = angles.to(device)
        labels = labels.to(device).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad()

        outputs = model(inputs, angles)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, angles, labels in dataloader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)


def train_model(
    model, train_loader, val_loader, epochs=50, patience=10, lr=1e-3, device="cuda"
):
    """
    Trains the model with Early Stopping.
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}"
        )

        # Early Stopping Logic
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model, best_loss


def predict_with_tta(model, dataloader, device="cuda"):
    """
    Generates predictions using Test-Time Augmentation (Original, H-Flip, V-Flip).
    """
    model.eval()
    predictions = []
    ids = []

    with torch.no_grad():
        for inputs, angles, batch_ids in dataloader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            # TTA 1: Original
            out1 = model(inputs, angles)

            # TTA 2: Horizontal Flip
            inputs_h = torch.flip(inputs, [3])
            out2 = model(inputs_h, angles)

            # TTA 3: Vertical Flip
            inputs_v = torch.flip(inputs, [2])
            out3 = model(inputs_v, angles)

            # Average predictions
            avg_out = (out1 + out2 + out3) / 3.0

            predictions.extend(avg_out.cpu().numpy().flatten().tolist())
            ids.extend(batch_ids)

    return ids, predictions

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from library.utils import get_device, set_seed


class SimpleCNN(nn.Module):
    """
    Simple 4-layer CNN with Global Max Pooling.
    Optimized for small, sparse radar datasets (Cite Lesson 00031).
    Structure: 64 -> 128 -> 128 -> 128 filters (Cite Lesson 00026).
    """

    def __init__(self, dropout_rate=0.5):
        super(SimpleCNN, self).__init__()

        # Block 1: 3 -> 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)

        # Block 2: 64 -> 128
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        # Block 3: 128 -> 128
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Block 4: 128 -> 128
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Global Max Pooling (Cite Lesson 00005, 00007)
        self.global_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier
        # 128 features + 1 angle = 129 inputs
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

    def forward(self, x, angle):
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool(x)

        # Global Max Pooling
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten -> (B, 128)

        # Fusion with Incidence Angle
        if angle.dim() == 1:
            angle = angle.view(-1, 1)

        x = torch.cat([x, angle], dim=1)  # -> (B, 129)

        # Classification Head
        # Dropout applied AFTER the first dense layer (Cite Lesson 00017)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


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

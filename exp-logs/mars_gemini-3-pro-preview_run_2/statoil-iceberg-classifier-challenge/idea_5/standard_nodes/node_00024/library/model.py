import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import copy
from library.utils import get_device, seed_everything
from library.data_loader import get_dataloaders

# ==========================================
# 1. Model Architecture
# ==========================================


class ResidualBlock(nn.Module):
    """
    Standard Residual Block with two 3x3 convolutions.
    Includes a skip connection with an optional 1x1 convolution for channel adaptation.
    """

    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class DRPPN(nn.Module):
    """
    Deep Residual Peak-Preserving Network.
    Features a visual branch with MaxPooling to preserve signal peaks and a metadata branch.
    """

    def __init__(self):
        super(DRPPN, self).__init__()

        # --- Visual Branch ---
        # Stem: Stride 1 to preserve initial spatial resolution (75x75)
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 75x75 -> 37x37
        self.layer1 = ResidualBlock(32, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2: 37x37 -> 18x18
        self.layer2 = ResidualBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3: 18x18 -> 9x9
        self.layer3 = ResidualBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 4: 9x9 -> 4x4
        self.layer4 = ResidualBlock(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # --- Metadata Branch ---
        self.meta_fc = nn.Sequential(
            nn.Linear(1, 32), nn.BatchNorm1d(32), nn.ReLU(inplace=True)
        )

        # --- Fusion Head ---
        # Flattened Visual: 256 channels * 4 * 4 = 4096
        # Metadata: 32
        # Total: 4128
        self.fusion = nn.Sequential(
            nn.Linear(4096 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 1),  # Output logits
        )

    def forward(self, x_img, x_angle):
        # Visual Processing
        v = self.stem(x_img)
        v = self.pool1(self.layer1(v))
        v = self.pool2(self.layer2(v))
        v = self.pool3(self.layer3(v))
        v = self.pool4(self.layer4(v))

        # Flatten: (N, 4096)
        v = v.view(v.size(0), -1)

        # Metadata Processing
        # Ensure angle is (N, 1)
        if x_angle.dim() == 1:
            x_angle = x_angle.unsqueeze(1)
        m = self.meta_fc(x_angle)

        # Fusion
        combined = torch.cat((v, m), dim=1)
        out = self.fusion(combined)
        return out


# ==========================================
# 2. Training Logic
# ==========================================


def train_model(model, train_loader, val_loader, epochs=50, patience=25, device=None):
    """
    Training loop with Early Stopping and Scheduler.
    """
    if device is None:
        device = get_device()

    model = model.to(device)

    # "Low and Slow" optimization
    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )
    criterion = nn.BCEWithLogitsLoss()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    epochs_no_improve = 0

    print(f"Starting training on {device} for {epochs} epochs...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0

        for inputs, angles, labels in train_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)
            labels = labels.to(device).unsqueeze(1)  # (N, 1)

            optimizer.zero_grad()

            outputs = model(inputs, angles)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, angles, labels in val_loader:
                inputs = inputs.to(device)
                angles = angles.to(device)
                labels = labels.to(device).unsqueeze(1)

                outputs = model(inputs, angles)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_loss / len(val_loader.dataset)

        # Print metrics with full precision
        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {epoch_loss:.6f} - Val Loss: {epoch_val_loss:.6f}"
        )

        # Scheduler Step
        scheduler.step(epoch_val_loss)

        # Early Stopping Check
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Best Validation Loss: {best_loss:.6f}")

    # Load best weights
    model.load_state_dict(best_model_wts)
    return model


# ==========================================
# 3. Inference and Submission
# ==========================================


def predict_and_submit(
    model, test_loader, ids_test, output_dir="./submission", device=None
):
    """
    Generates predictions and saves to CSV.
    """
    if device is None:
        device = get_device()

    model.eval()
    model = model.to(device)

    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs, angles in test_loader:
            inputs = inputs.to(device)
            angles = angles.to(device)

            # Forward pass
            logits = model(inputs, angles)
            probs = torch.sigmoid(logits)

            predictions.extend(probs.cpu().numpy().flatten().tolist())

    # Ensure alignment
    if len(predictions) != len(ids_test):
        print(
            f"Warning: Prediction count ({len(predictions)}) matches ID count ({len(ids_test)})?"
        )

    # Create DataFrame
    df_sub = pd.DataFrame({"id": ids_test, "is_iceberg": predictions})

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "submission.csv")
    df_sub.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")


# ==========================================
# 4. Main Execution Helper
# ==========================================


def run_training_process(epochs=100, batch_size=32):
    """
    Orchestrates the entire pipeline: Data loading, Training, Prediction.
    """
    seed_everything(42)
    device = get_device()

    # 1. Get Data
    print("Loading data...")
    train_loader, val_loader, test_loader, ids_test = get_dataloaders(
        batch_size=batch_size
    )

    # 2. Initialize Model
    model = DRPPN()

    # 3. Train
    trained_model = train_model(
        model, train_loader, val_loader, epochs=epochs, patience=30, device=device
    )

    # 4. Predict
    predict_and_submit(trained_model, test_loader, ids_test, device=device)

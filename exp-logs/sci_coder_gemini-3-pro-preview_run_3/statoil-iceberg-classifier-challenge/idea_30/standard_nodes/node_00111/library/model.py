import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


class SEModule(nn.Module):
    """
    Squeeze-and-Excitation Module with Global Average Pooling.
    Acts as a channel-wise attention mechanism.
    """

    def __init__(self, channels, reduction=16):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: Global Average Pooling
        y = self.avg_pool(x).view(b, c)
        # Excitation: Adaptive Recalibration
        y = self.fc(y).view(b, c, 1, 1)
        # Scale
        return x * y


class IcebergCNN(nn.Module):
    """
    Selective Hierarchical Hybrid-SE CNN.

    Structure:
    - 4-Stage Plain CNN Backbone (Conv-BN-Leaky-SE-Pool)
    - Selective Readout: Global Max Pooling on Stage 3 and Stage 4
    - Fusion: Concat(Pool3, Pool4, Angle)
    - Classifier: Single Hidden Layer
    """

    def __init__(self):
        super(IcebergCNN, self).__init__()

        # Stage 1: 3 -> 64 (75x75 -> 37x37)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(64),
            nn.MaxPool2d(2, 2),
        )

        # Stage 2: 64 -> 128 (37x37 -> 18x18)
        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 3: 128 -> 128 (18x18 -> 9x9)
        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128),
            nn.MaxPool2d(2, 2),
        )

        # Stage 4: 128 -> 128 (9x9 -> 4x4)
        self.stage4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            SEModule(128),
            nn.MaxPool2d(2, 2),
        )

        # Pooling for Readout
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)

        # Classifier
        # Input Dimension: 128 (Stage 3) + 128 (Stage 4) + 1 (Angle) = 257
        self.classifier = nn.Sequential(
            nn.Linear(257, 256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

        self._init_weights()

    def _init_weights(self):
        # Kaiming Uniform Initialization adapted for LeakyReLU
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu"
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_img, x_angle):
        # Backbone Forward Pass
        s1 = self.stage1(x_img)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)

        # Selective Hierarchical Readout
        # Flatten Stage 3 features (Global Max)
        p3 = self.global_max_pool(s3).view(s3.size(0), -1)
        # Flatten Stage 4 features (Global Max)
        p4 = self.global_max_pool(s4).view(s4.size(0), -1)

        # Reshape Angle
        angle = x_angle.view(-1, 1)

        # Feature Fusion
        fused = torch.cat([p3, p4, angle], dim=1)

        # Classification
        out = self.classifier(fused)

        return out


def train_model(
    train_loader,
    val_loader,
    device,
    epochs=75,
    patience=12,
    lr=1e-3,
    save_path="model_best.pth",
):
    """
    Trains the IcebergCNN model using AdamW and Early Stopping.
    """
    model = IcebergCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()

    # AdamW with Decoupled Weight Decay (Constant LR)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    best_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on {device}...")

    for epoch in range(epochs):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0

        for (imgs, angles), labels in train_loader:
            imgs, angles, labels = imgs.to(device), angles.to(device), labels.to(device)
            labels = labels.view(-1, 1)

            optimizer.zero_grad()
            outputs = model(imgs, angles)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for (imgs, angles), labels in val_loader:
                imgs, angles, labels = (
                    imgs.to(device),
                    angles.to(device),
                    labels.to(device),
                )
                labels = labels.view(-1, 1)

                outputs = model(imgs, angles)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)

                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
        )

        # --- Early Stopping ---
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    print(f"Training complete. Best Val Loss: {best_loss:.6f}")

    # Load best weights
    model.load_state_dict(torch.load(save_path))
    return model


def predict(model, test_loader, device):
    """
    Generates probabilities for the test set.
    """
    model.eval()
    preds_list = []
    with torch.no_grad():
        for imgs, angles in test_loader:
            imgs, angles = imgs.to(device), angles.to(device)
            outputs = model(imgs, angles)
            probs = torch.sigmoid(outputs)
            preds_list.extend(probs.cpu().numpy().flatten().tolist())
    return np.array(preds_list)


def generate_submission(
    model, test_loader, device, output_path="./submission/submission.csv"
):
    """
    Generates predictions and saves them to a CSV file in the required format.
    """
    print(f"Generating submission to {output_path}...")
    preds = predict(model, test_loader, device)
    ids = test_loader.dataset.ids

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame({"id": ids, "is_iceberg": preds})

    df.to_csv(output_path, index=False)
    print("Submission saved.")

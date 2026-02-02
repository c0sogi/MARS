import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import roc_auc_score
from library import config, data


# ==========================================
# Reproducibility
# ==========================================
def set_seed(seed=config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# Model Architecture
# ==========================================
class Stabilized25DNet(nn.Module):
    """
    2.5D Convolutional Neural Network with a Stabilized Projection Stem.

    Structure:
    1. Stem: Projects high-density input (128 channels) -> 64 channels.
    2. Backbone: EfficientNet-B0 (First layer modified for 64 channels).
    3. Head: Linear classification layer.
    """

    def __init__(self):
        super(Stabilized25DNet, self).__init__()

        # 1. Stabilized Projection Stem
        self.stem = nn.Sequential(
            nn.Conv2d(
                in_channels=config.INPUT_CHANNELS,  # 128
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Initialize Stem with Kaiming Normal
        self._init_stem()

        # 2. Backbone: EfficientNet-B0
        # Load pre-trained weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

        # Modify the first convolutional layer to accept 64 channels
        # efficientnet_b0.features[0][0] is the first Conv2d
        original_first_conv = self.backbone.features[0][0]

        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=64,
            out_channels=original_first_conv.out_channels,
            kernel_size=original_first_conv.kernel_size,
            stride=original_first_conv.stride,
            padding=original_first_conv.padding,
            bias=False,
        )

        # 3. Head
        # Replace the default classifier
        # EfficientNet-B0 output features: 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True), nn.Linear(1280, 1)
        )

    def _init_stem(self):
        for m in self.stem.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, 128, 256, 256)

        # Pass through Stem
        x = self.stem(x)  # -> (B, 64, 256, 256)

        # Pass through Backbone (Features -> Pool -> Flatten -> Classifier)
        logits = self.backbone(x)

        return logits


# ==========================================
# Training Logic
# ==========================================
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device).unsqueeze(1)  # (B) -> (B, 1)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # Store predictions for AUC calculation
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_preds.append(probs)
        all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)

            logits = model(inputs)
            loss = criterion(logits, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    all_preds = np.concatenate(all_preds).flatten()
    all_targets = np.concatenate(all_targets).flatten()

    try:
        auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        auc = 0.5

    return epoch_loss, auc


def train_model(epochs=config.EPOCHS, batch_size=config.BATCH_SIZE):
    set_seed()
    device = config.DEVICE

    # Get DataLoaders
    train_loader, val_loader, _, _ = data.get_dataloaders(batch_size=batch_size)

    # Initialize Model, Optimizer, Loss
    model = Stabilized25DNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()

    print(f"Starting training on {device} for {epochs} epochs...")

    best_val_auc = 0.0
    patience_counter = 0

    for epoch in range(epochs):
        train_loss, train_auc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.6f} | Train AUC: {train_auc:.6f} | "
            f"Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
        )

        # Early Stopping & Checkpointing
        if val_auc > best_val_auc + config.MIN_DELTA:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    print(f"Training complete. Best Val AUC: {best_val_auc:.6f}")


# ==========================================
# Inference Logic
# ==========================================
def predict(batch_size=config.BATCH_SIZE):
    set_seed()
    device = config.DEVICE

    # Get Test DataLoader
    _, _, test_loader, test_ids = data.get_dataloaders(batch_size=batch_size)

    # Load Model
    model = Stabilized25DNet().to(device)
    if os.path.exists(config.MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    else:
        print("Warning: No trained model found. Predictions will be random/untrained.")

    model.eval()
    predictions = []

    print("Generating predictions...")
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            predictions.extend(probs)

    # Save Submission
    submission_df = pd.DataFrame({"BraTS21ID": test_ids, "MGMT_value": predictions})

    os.makedirs(config.SUBMISSION_DIR, exist_ok=True)
    submission_df.to_csv(config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {config.SUBMISSION_PATH}")

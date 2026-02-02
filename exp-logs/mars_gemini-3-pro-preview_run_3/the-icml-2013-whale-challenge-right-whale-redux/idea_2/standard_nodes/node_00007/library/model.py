import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from library.config import Config
from library.dataset import WhaleDataset
from library.utils import mixup_data, mixed_criterion, calculate_roc_auc, set_seed


class SEBasicBlock(nn.Module):
    """
    Residual Block with Squeeze-and-Excitation (SE) module.
    """

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(SEBasicBlock, self).__init__()
        # Standard ResNet Convolution Path
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        # Squeeze-and-Excitation Path
        # Ensure bottleneck dimension is at least 1
        reduced_dim = max(1, planes // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(planes, reduced_dim),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_dim, planes),
            nn.Sigmoid(),
        )

        # Shortcut Connection
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply SE Attention
        b, c, _, _ = out.size()
        y = self.se(out).view(b, c, 1, 1)
        out = out * y

        # Residual Connection
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEResNet(nn.Module):
    """
    Deep Residual Network with Channel Attention (SE-ResNet).
    Adapted for 1-channel Log-Mel Spectrogram inputs.
    """

    def __init__(self):
        super(SEResNet, self).__init__()
        self.inplanes = Config.BASE_CHANNELS

        # Initial Convolution
        # Input: (B, 1, 128, 125) -> Output: (B, 32, 64, 63)
        self.conv1 = nn.Conv2d(
            Config.IN_CHANNELS,
            Config.BASE_CHANNELS,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(Config.BASE_CHANNELS)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual Layers
        # Layer 1: 32 channels
        self.layer1 = self._make_layer(Config.BASE_CHANNELS, 2, stride=1)
        # Layer 2: 64 channels
        self.layer2 = self._make_layer(Config.BASE_CHANNELS * 2, 2, stride=2)
        # Layer 3: 128 channels
        self.layer3 = self._make_layer(Config.BASE_CHANNELS * 4, 2, stride=2)
        # Layer 4: 256 channels
        self.layer4 = self._make_layer(Config.BASE_CHANNELS * 8, 2, stride=2)

        # Classifier Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(Config.BASE_CHANNELS * 8, Config.NUM_CLASSES)

        # Weight Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, planes, blocks, stride):
        layers = []
        layers.append(SEBasicBlock(self.inplanes, planes, stride))
        self.inplanes = planes * SEBasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(SEBasicBlock(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # Return logits (Sigmoid applied in loss/inference)
        return x


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        # Mixup Augmentation
        data, target_a, target_b, lam = mixup_data(
            data, target, Config.MIXUP_ALPHA, device
        )

        # Forward pass
        output = model(data)

        # Mixed Loss
        loss = mixed_criterion(
            criterion, output, target_a.view(-1, 1), target_b.view(-1, 1), lam
        )

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        # Store predictions for rough epoch metric (using primary target)
        with torch.no_grad():
            probs = torch.sigmoid(output)
            all_preds.extend(probs.cpu().numpy())
            # For mixup, we can't easily calc exact accuracy, but we track primary label
            all_targets.extend(target_a.cpu().numpy())

    epoch_loss = running_loss / len(loader)
    # Note: AUC here is noisy due to mixup labels, but useful for sanity check
    try:
        epoch_auc = calculate_roc_auc(all_targets, all_preds)
    except:
        epoch_auc = 0.5

    print(
        f"Epoch {epoch+1}/{Config.EPOCHS} - Train Loss: {epoch_loss:.6f} - Train AUC: {epoch_auc:.6f}"
    )
    return epoch_loss


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target.view(-1, 1))

            running_loss += loss.item()

            probs = torch.sigmoid(output)
            all_preds.extend(probs.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_auc = calculate_roc_auc(all_targets, all_preds)

    print(f"Validation - Loss: {val_loss:.6f} - AUC: {val_auc:.6f}")
    return val_loss, val_auc


def predict(model, loader, device):
    model.eval()
    results = []

    with torch.no_grad():
        for i in range(len(loader)):
            # Dataset returns (spec, label), but we also need clip name
            # We access dataset directly or iterate loader.
            # Loader batching makes accessing clip names tricky unless we modify collate or dataset.
            # Simpler approach: iterate loader, keep track of indices, ask dataset for names.

            data, _ = loader.dataset[i]
            clip_name = loader.dataset.get_clip_name(i)

            # Add batch dimension
            data = data.unsqueeze(0).to(device)

            output = model(data)
            prob = torch.sigmoid(output).item()

            results.append({"clip": clip_name, "probability": prob})

            if (i + 1) % 1000 == 0:
                print(f"Processed {i+1}/{len(loader.dataset)} test samples...")

    return pd.DataFrame(results)


def main():
    Config.setup()
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # --- Data Loading ---
    print("Initializing Datasets...")
    train_dataset = WhaleDataset(split="train", load_cached_data=True)
    val_dataset = WhaleDataset(split="val", load_cached_data=True)

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

    # --- Model Setup ---
    print("Initializing Model...")
    model = SEResNet().to(device)

    # Optimizer & Scheduler
    optimizer = optim.Adam(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)

    # Loss Function (Weighted BCE)
    pos_weight = torch.tensor([Config.POS_WEIGHT]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # --- Training Loop ---
    best_auc = 0.0
    patience_counter = 0
    best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Checkpointing
        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New Best AUC! Model saved to {best_model_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}")

        print(f"Time: {time.time() - start_time:.2f}s")
        print("-" * 30)

        # Early Stopping
        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    # --- Inference ---
    print("Starting Inference on Test Set...")

    # Load Best Model
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print("Loaded best model.")
    else:
        print("Warning: No best model found. Using current model.")

    # Test Dataset (No DataLoader shuffling, batch size 1 for simplicity in mapping names)
    test_dataset = WhaleDataset(split="test", load_cached_data=True)
    # Note: Using batch_size=1 to easily map clip names in the predict function loop
    # Ideally, we would batch this and pass names through, but the dataset API provided
    # has get_clip_name(idx).

    # Custom prediction loop to handle mapping
    model.eval()
    results = []

    # Process in batches for speed, but map indices manually
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
    )

    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(test_loader):
            data = data.to(device)
            output = model(data)
            probs = torch.sigmoid(output).cpu().numpy().flatten()

            # Calculate global indices
            start_idx = batch_idx * Config.BATCH_SIZE
            for i, prob in enumerate(probs):
                global_idx = start_idx + i
                if global_idx < len(test_dataset):
                    clip_name = test_dataset.get_clip_name(global_idx)
                    results.append({"clip": clip_name, "probability": prob})

    df_submission = pd.DataFrame(results)

    # Save Submission
    print(f"Saving submission to {Config.SUBMISSION_FILE}...")
    df_submission.to_csv(Config.SUBMISSION_FILE, index=False)
    print("Done.")


# Note: The main() function is defined but not called at the top level
# to comply with the instruction: "Only implement the module class/functions."
# The evaluation environment is expected to import this module and call main()
# or use the classes defined herein.

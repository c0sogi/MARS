import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from library.dataset import get_dataloaders, get_test_ids
from library.utils import seed_everything, AverageMeter, calculate_roc_auc

# --- Configuration ---
BATCH_SIZE = 64
NUM_WORKERS = 2
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
N_SEEDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Components ---


class BasicBlock(nn.Module):
    """
    Standard Residual Block with 2 convolutions.
    Cite solution_lesson_node_00063: Prefer 1x1 convolutions for projection shortcuts.
    """

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SimpleResNet(nn.Module):
    """
    Lightweight Custom ResNet tailored for 32x32 images.
    Cite solution_lesson_node_00005: Preserve Spatial Resolution, prefer GAP.
    Cite solution_lesson_node_00013: Avoid architectural complexity (SE/Attention).
    """

    def __init__(self):
        super(SimpleResNet, self).__init__()
        self.in_planes = 32

        # Stem: 32x32 image, 3x3 conv, no maxpool to preserve resolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Stages
        # Cite solution_lesson_node_00049: Prioritize model width (32->128).
        self.layer1 = self._make_layer(32, 2, stride=1)  # 32x32
        self.layer2 = self._make_layer(64, 2, stride=2)  # 16x16
        self.layer3 = self._make_layer(128, 2, stride=2)  # 8x8

        self.linear = nn.Linear(128, 1)

    def _make_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(BasicBlock(self.in_planes, planes, stride))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


# --- Training & Execution ---


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses = AverageMeter()

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    return losses.avg


def validate(model, loader, criterion, device):
    model.eval()
    losses = AverageMeter()
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            losses.update(loss.item(), images.size(0))
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = calculate_roc_auc(targets, preds)
    return losses.avg, auc


def predict(model, loader, device):
    model.eval()
    preds = []

    with torch.no_grad():
        for images in loader:
            images = images.to(device)

            # TTA: Original
            out = torch.sigmoid(model(images))

            # TTA: Horizontal Flip
            out_h = torch.sigmoid(model(torch.flip(images, [3])))

            # TTA: Vertical Flip
            out_v = torch.sigmoid(model(torch.flip(images, [2])))

            # Average
            p = (out + out_h + out_v) / 3.0
            preds.extend(p.cpu().numpy().flatten())

    return np.array(preds)


def run_experiment():
    # Ensure directories exist
    os.makedirs("./submission", exist_ok=True)
    os.makedirs("./working/idea_29", exist_ok=True)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, load_cached_data=True
    )
    test_ids = get_test_ids(load_cached_data=True)

    final_preds = np.zeros(len(test_ids))

    # Homogeneous Seed Averaging
    for seed in range(N_SEEDS):
        print(f"Training Seed {seed}...")
        seed_everything(seed)

        model = SimpleResNet().to(DEVICE)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        best_model_path = f"./working/idea_29/model_seed_{seed}.pth"

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
            )

        # Load best model for inference
        if os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

        # Predict with TTA
        seed_preds = predict(model, test_loader, DEVICE)
        final_preds += seed_preds

    # Average predictions across seeds
    final_preds /= N_SEEDS

    # Save Submission
    df_sub = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})
    df_sub.to_csv("./submission/submission.csv", index=False)
    print("Submission saved to ./submission/submission.csv")


# Execute experiment
run_experiment()

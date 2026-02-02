import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
from sklearn.metrics import roc_auc_score

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_26"
SUBMISSION_DIR = "./submission"
IMG_SIZE = 32
BATCH_SIZE = 64
EPOCHS = 20
SEEDS = [0, 1, 2, 3, 4]
NUM_WORKERS = 2

# Ensure directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# --- Utilities ---
def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    """Returns the appropriate device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model Components ---


class CoordinateAttention(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.Hardswish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x
        n, c, h, w = x.size()

        # Pool
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # Expand
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_h * a_w
        return out


class ResNeXtBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, cardinality=32, reduction=16):
        super(ResNeXtBlock, self).__init__()
        group_width = planes // 2  # Bottleneck width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)

        self.conv2 = nn.Conv2d(
            group_width,
            group_width,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(group_width)

        self.conv3 = nn.Conv2d(group_width, planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.attn = CoordinateAttention(planes, reduction=reduction)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.attn(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class WideCoordinateResNeXt(nn.Module):
    def __init__(self, cardinality=32):
        super(WideCoordinateResNeXt, self).__init__()
        # Input 32x32
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        # Stage 1: 32x32 -> 32x32. Channels: 64
        self.layer1 = self._make_layer(64, 64, 2, stride=1, cardinality=cardinality)

        # Stage 2: 32x32 -> 16x16. Channels: 128
        self.layer2 = self._make_layer(64, 128, 2, stride=2, cardinality=cardinality)

        # Stage 3: 16x16 -> 8x8. Channels: 256
        self.layer3 = self._make_layer(128, 256, 2, stride=2, cardinality=cardinality)

        # Head
        self.fc = nn.Linear(128 + 256, 1)

    def _make_layer(self, in_planes, planes, num_blocks, stride, cardinality):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(
                ResNeXtBlock(in_planes, planes, stride=s, cardinality=cardinality)
            )
            in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))  # 32x32

        out1 = self.layer1(out)  # 32x32
        out2 = self.layer2(out1)  # 16x16
        out3 = self.layer3(out2)  # 8x8

        # Multi-Scale Aggregation
        gap2 = F.adaptive_avg_pool2d(out2, 1).view(out2.size(0), -1)  # 128
        gap3 = F.adaptive_avg_pool2d(out3, 1).view(out3.size(0), -1)  # 256

        combined = torch.cat([gap2, gap3], dim=1)
        return self.fc(combined)


# --- Dataset ---


class CactusDataset(Dataset):
    def __init__(self, metadata_path, transform=None, is_test=False):
        self.df = pd.read_csv(metadata_path)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Resolve path
        img_path = os.path.join(INPUT_DIR, row["file_path"])

        image = cv2.imread(img_path)
        if image is None:
            # Fallback for robustness
            image = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Normalize
        image = image.astype(np.float32) / 255.0

        # Augmentation (Manual implementation for simplicity/speed)
        if self.transform:
            if random.random() > 0.5:
                image = cv2.flip(image, 1)  # Horizontal
            if random.random() > 0.5:
                image = cv2.flip(image, 0)  # Vertical

        # To Tensor (C, H, W)
        image = torch.from_numpy(image.transpose(2, 0, 1))

        if self.is_test:
            return image, row["id"]
        else:
            return image, torch.tensor(row["has_cactus"], dtype=torch.float32)


# --- Training & Evaluation ---


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    auc = roc_auc_score(targets, preds)
    return running_loss / len(loader.dataset), auc


def predict_tta(model, image_tensor, device):
    """Predicts with TTA: Original, HFlip, VFlip."""
    model.eval()
    # Create batch of 3: Original, HFlip, VFlip
    img_h = torch.flip(image_tensor, [2])
    img_v = torch.flip(image_tensor, [1])

    batch = torch.stack([image_tensor, img_h, img_v]).to(device)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits)

    return probs.mean().item()


# --- Main Execution ---


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Data Loaders
    train_meta = os.path.join(METADATA_DIR, "train_metadata.csv")
    val_meta = os.path.join(METADATA_DIR, "val_metadata.csv")
    test_meta = os.path.join(METADATA_DIR, "test_metadata.csv")

    # We use a simple transform flag in dataset
    train_dataset = CactusDataset(train_meta, transform=True)
    val_dataset = CactusDataset(val_meta, transform=False)
    test_dataset = CactusDataset(test_meta, transform=False, is_test=True)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    # Test loader batch size 1 for simple TTA loop or larger for efficiency?
    # Let's do batch processing for test if possible, but TTA logic is easier per image or custom collate.
    # For speed, we'll iterate test set and do TTA inside.

    test_ids = test_dataset.df["id"].values
    # Placeholder for ensemble predictions
    final_preds = np.zeros(len(test_dataset))

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        model = WideCoordinateResNeXt().to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_auc = 0.0
        best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_auc = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val AUC: {val_auc:.6f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        # Load best model for inference
        model.load_state_dict(torch.load(best_model_path, map_location=device))

        # Inference on Test Set
        print(f"Generating predictions for Seed {seed}...")
        seed_preds = []

        # Manual TTA Loop
        # To optimize, we can create a DataLoader that returns images
        test_loader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, num_workers=NUM_WORKERS
        )

        for img, _ in test_loader:
            img = img.squeeze(0)  # (C, H, W)
            prob = predict_tta(model, img, device)
            seed_preds.append(prob)

        final_preds += np.array(seed_preds)

    # Average predictions
    final_preds /= len(SEEDS)

    # Save Submission
    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    out_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"\nSubmission saved to {out_path}")
    print(sub_df.head())


if __name__ == "__main__":
    main()

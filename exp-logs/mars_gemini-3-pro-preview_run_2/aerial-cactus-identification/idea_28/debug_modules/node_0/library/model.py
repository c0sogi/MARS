import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from library.layers import Res2NeXtBlock, GeM
from library.utils import set_seed, get_device

# --- Configuration ---
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_28"
SUBMISSION_DIR = "./submission"
IMG_SIZE = 32
BATCH_SIZE = 128
EPOCHS = 20
NUM_SEEDS = 5
BASE_LR = 1e-3
EARLY_STOPPING_PATIENCE = 5


# --- Data Loading & Caching ---
def load_data(load_cached_data=True):
    """
    Loads data from disk or cache.
    Returns: (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids)
    """
    os.makedirs(WORKING_DIR, exist_ok=True)

    cache_files = {
        "train_imgs": os.path.join(WORKING_DIR, "train_imgs.npy"),
        "train_labels": os.path.join(WORKING_DIR, "train_labels.npy"),
        "val_imgs": os.path.join(WORKING_DIR, "val_imgs.npy"),
        "val_labels": os.path.join(WORKING_DIR, "val_labels.npy"),
        "test_imgs": os.path.join(WORKING_DIR, "test_imgs.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    cache_exists = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and cache_exists:
        print("Loading data from cache...")
        train_imgs = np.load(cache_files["train_imgs"])
        train_labels = np.load(cache_files["train_labels"])
        val_imgs = np.load(cache_files["val_imgs"])
        val_labels = np.load(cache_files["val_labels"])
        test_imgs = np.load(cache_files["test_imgs"])
        test_ids = np.load(cache_files["test_ids"])
        return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids)

    print("Cache missing or reload requested. Processing data from scratch...")

    # Load Metadata
    train_meta = pd.read_csv(os.path.join(METADATA_DIR, "train_metadata.csv"))
    val_meta = pd.read_csv(os.path.join(METADATA_DIR, "val_metadata.csv"))
    test_meta = pd.read_csv(os.path.join(METADATA_DIR, "test_metadata.csv"))

    def load_images(meta_df):
        imgs = []
        ids = []
        labels = []
        for _, row in meta_df.iterrows():
            # file_path is relative to INPUT_DIR
            path = os.path.join(INPUT_DIR, row["file_path"])
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            imgs.append(img)
            ids.append(row["id"])
            if "has_cactus" in row:
                labels.append(row["has_cactus"])
        return np.array(imgs), np.array(ids), np.array(labels)

    train_imgs, _, train_labels = load_images(train_meta)
    val_imgs, _, val_labels = load_images(val_meta)
    test_imgs, test_ids, _ = load_images(test_meta)

    # Save to cache
    np.save(cache_files["train_imgs"], train_imgs)
    np.save(cache_files["train_labels"], train_labels)
    np.save(cache_files["val_imgs"], val_imgs)
    np.save(cache_files["val_labels"], val_labels)
    np.save(cache_files["test_imgs"], test_imgs)
    np.save(cache_files["test_ids"], test_ids)

    return (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids)


# --- Dataset ---
class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=False):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Augmentation (Light: Flips only)
        if self.transform:
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=1)  # H-Flip
            if np.random.rand() > 0.5:
                img = np.flip(img, axis=0)  # V-Flip

        # Normalize to [0, 1] and CHW
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW

        img_tensor = torch.from_numpy(img.copy())

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img_tensor, label
        return img_tensor


# --- Model ---
class WideAntiAliasedCoordRes2NeXt(nn.Module):
    def __init__(self, num_classes=1):
        super(WideAntiAliasedCoordRes2NeXt, self).__init__()

        # Stem: 3x3 Conv, Stride 1 (No aggressive downsampling for 32x32)
        self.in_planes = 64
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 64 -> 256 (Expansion 4)
        self.layer1 = self._make_layer(Res2NeXtBlock, planes=64, blocks=3, stride=1)

        # Stage 2: 256 -> 512 (Expansion 4)
        self.layer2 = self._make_layer(Res2NeXtBlock, planes=128, blocks=3, stride=2)

        # Stage 3: 512 -> 1024 (Expansion 4)
        self.layer3 = self._make_layer(Res2NeXtBlock, planes=256, blocks=3, stride=2)

        # Multi-Scale Head
        self.gem2 = GeM()
        self.gem3 = GeM()

        # Final FC
        # Stage 2 output channels: 128 * 4 = 512
        # Stage 3 output channels: 256 * 4 = 1024
        self.fc = nn.Linear(512 + 1024, num_classes)

    def _make_layer(self, block, planes, blocks, stride):
        layers = []
        # First block handles stride and channel expansion
        layers.append(
            block(
                self.in_planes,
                planes,
                stride=stride,
                cardinality=32,
                scales=4,
                base_width=4,
            )
        )
        self.in_planes = planes * 4  # Expansion is 4

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_planes,
                    planes,
                    stride=1,
                    cardinality=32,
                    scales=4,
                    base_width=4,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)

        x = self.layer1(x)

        f2 = self.layer2(x)  # Stage 2 features (16x16)
        f3 = self.layer3(f2)  # Stage 3 features (8x8)

        # Multi-Scale Aggregation
        p2 = self.gem2(f2).view(f2.size(0), -1)
        p3 = self.gem3(f3).view(f3.size(0), -1)

        out = torch.cat([p2, p3], dim=1)
        out = self.fc(out)

        return out


# --- Training Engine ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def train_instance(seed, train_data, val_data):
    set_seed(seed)
    device = get_device()

    train_imgs, train_labels = train_data
    val_imgs, val_labels = val_data

    train_dataset = CactusDataset(train_imgs, train_labels, transform=True)
    val_dataset = CactusDataset(val_imgs, val_labels, transform=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = WideAntiAliasedCoordRes2NeXt().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = os.path.join(WORKING_DIR, f"model_seed_{seed}.pth")

    print(f"\nStarting training for Seed {seed}")

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    return best_model_path


# --- Inference ---
def predict_with_tta(model, images, device):
    model.eval()
    # TTA: Original, H-Flip, V-Flip

    # Prepare inputs
    img_orig = images
    img_h = torch.flip(images, [3])
    img_v = torch.flip(images, [2])

    with torch.no_grad():
        out_orig = torch.sigmoid(model(img_orig))
        out_h = torch.sigmoid(model(img_h))
        out_v = torch.sigmoid(model(img_v))

    # Average
    avg_preds = (out_orig + out_h + out_v) / 3.0
    return avg_preds.cpu().numpy()


def generate_submission(test_imgs, test_ids, model_paths):
    device = get_device()
    test_dataset = CactusDataset(test_imgs, transform=False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Load all models
    models = []
    for path in model_paths:
        m = WideAntiAliasedCoordRes2NeXt().to(device)
        m.load_state_dict(torch.load(path, map_location=device))
        m.eval()
        models.append(m)

    all_preds = []

    print("Generating predictions with TTA...")
    for images in test_loader:
        images = images.to(device)

        batch_preds = []
        for model in models:
            p = predict_with_tta(model, images, device)
            batch_preds.append(p)

        # Average across models
        batch_preds_avg = np.mean(batch_preds, axis=0)
        all_preds.append(batch_preds_avg)

    final_preds = np.concatenate(all_preds, axis=0).flatten()

    # Create submission dataframe
    df = pd.DataFrame({"id": test_ids, "has_cactus": final_preds})

    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    sub_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# --- Main Pipeline ---
def run_experiment(load_cached_data=True):
    print("Initializing Experiment: Custom Wide Anti-Aliased Coordinate-Res2NeXt")

    # 1. Load Data
    (train_imgs, train_labels), (val_imgs, val_labels), (test_imgs, test_ids) = (
        load_data(load_cached_data)
    )

    # 2. Train Ensemble
    model_paths = []
    for seed in range(NUM_SEEDS):
        path = train_instance(seed, (train_imgs, train_labels), (val_imgs, val_labels))
        model_paths.append(path)

    # 3. Generate Submission
    generate_submission(test_imgs, test_ids, model_paths)
    print("Experiment Completed.")

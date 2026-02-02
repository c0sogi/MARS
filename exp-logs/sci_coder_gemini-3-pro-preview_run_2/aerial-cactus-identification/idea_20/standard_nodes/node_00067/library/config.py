import os
import random
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score

# ==========================================
# Configuration & Constants
# ==========================================
INPUT_DIR = "./input"
METADATA_DIR = "./metadata"
WORKING_DIR = "./working/idea_21"
SUBMISSION_DIR = os.path.join(WORKING_DIR, "submission")

# Ensure working directories exist
os.makedirs(WORKING_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# Hyperparameters
BATCH_SIZE = 64
EPOCHS = 15
LEARNING_RATE = 1e-3
SEEDS = [0, 1, 2, 3, 4]
CHANNELS = [16, 32, 64]
IMAGE_SIZE = 32
NUM_WORKERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# Data Processing & Caching
# ==========================================
def load_data(load_cached_data=True):
    """
    Loads data from metadata CSVs and images.
    Implements caching using .npy files in WORKING_DIR.
    """
    cache_files = {
        "train_img": os.path.join(WORKING_DIR, "train_images.npy"),
        "train_lbl": os.path.join(WORKING_DIR, "train_labels.npy"),
        "val_img": os.path.join(WORKING_DIR, "val_images.npy"),
        "val_lbl": os.path.join(WORKING_DIR, "val_labels.npy"),
        "test_img": os.path.join(WORKING_DIR, "test_images.npy"),
        "test_ids": os.path.join(WORKING_DIR, "test_ids.npy"),
    }

    # Check if all cache files exist
    all_cached = all(os.path.exists(p) for p in cache_files.values())

    if load_cached_data and all_cached:
        print("Loading data from cache...")
        data = {
            "train_images": np.load(cache_files["train_img"]),
            "train_labels": np.load(cache_files["train_lbl"]),
            "val_images": np.load(cache_files["val_img"]),
            "val_labels": np.load(cache_files["val_lbl"]),
            "test_images": np.load(cache_files["test_img"]),
            "test_ids": np.load(cache_files["test_ids"], allow_pickle=True),
        }
        return data

    print("Processing data from scratch...")

    # Helper to load images
    def process_split(metadata_path, is_test=False):
        df = pd.read_csv(metadata_path)
        images = []
        labels_or_ids = []

        for _, row in df.iterrows():
            # Construct full path: input_dir + relative_path_from_metadata
            img_path = os.path.join(INPUT_DIR, row["file_path"])
            img = cv2.imread(img_path)
            if img is None:
                continue

            # Keep in BGR or convert to RGB? PyTorch usually expects RGB.
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)

            if is_test:
                labels_or_ids.append(row["id"])
            else:
                labels_or_ids.append(row["has_cactus"])

        return np.array(images, dtype=np.uint8), np.array(labels_or_ids)

    # Process Train
    train_images, train_labels = process_split(
        os.path.join(METADATA_DIR, "train_metadata.csv")
    )
    # Process Val
    val_images, val_labels = process_split(
        os.path.join(METADATA_DIR, "val_metadata.csv")
    )
    # Process Test
    test_images, test_ids = process_split(
        os.path.join(METADATA_DIR, "test_metadata.csv"), is_test=True
    )

    # Save to cache
    np.save(cache_files["train_img"], train_images)
    np.save(cache_files["train_lbl"], train_labels)
    np.save(cache_files["val_img"], val_images)
    np.save(cache_files["val_lbl"], val_labels)
    np.save(cache_files["test_img"], test_images)
    np.save(cache_files["test_ids"], test_ids)

    return {
        "train_images": train_images,
        "train_labels": train_labels,
        "val_images": val_images,
        "val_labels": val_labels,
        "test_images": test_images,
        "test_ids": test_ids,
    }


class CactusDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Augmentation or Normalization
        # Images are uint8 (0-255). Transform should handle conversion to float/tensor.
        if self.transform:
            img = self.transform(img)
        else:
            # Default: ToTensor and Normalize
            img = torch.from_numpy(img.transpose((2, 0, 1))).float() / 255.0

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return img, label
        return img


# ==========================================
# Model Architecture: Custom Wide SE-ResNeXt
# ==========================================
class SEBlock(nn.Module):
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
        return x * y


class ResNeXtBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, stride=1, cardinality=32, downsample=None
    ):
        super(ResNeXtBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Grouped Convolution
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        out += identity
        out = self.relu(out)
        return out


class CustomWideSEResNeXt(nn.Module):
    def __init__(self, channels=CHANNELS, cardinality=CARDINALITY):
        super(CustomWideSEResNeXt, self).__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1: 32x32, channels[0]
        self.layer1 = self._make_layer(
            channels[0], channels[0], stride=1, cardinality=cardinality
        )

        # Stage 2: 16x16, channels[1]
        self.layer2 = self._make_layer(
            channels[0], channels[1], stride=2, cardinality=cardinality
        )

        # Stage 3: 8x8, channels[2]
        self.layer3 = self._make_layer(
            channels[1], channels[2], stride=2, cardinality=cardinality
        )

        # Head
        self.gap = nn.AdaptiveAvgPool2d(1)
        # Concatenation of Stage 2 and Stage 3 features
        self.fc = nn.Linear(channels[1] + channels[2], 1)

    def _make_layer(self, in_channels, out_channels, stride, cardinality):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

        # For ResNeXt, usually we have bottleneck expansion, but here we keep width consistent for simplicity
        # and match the "Wide" description by just using large channel counts.
        return ResNeXtBlock(in_channels, out_channels, stride, cardinality, downsample)

    def forward(self, x):
        x = self.stem(x)  # 32x32
        x1 = self.layer1(x)  # 32x32
        x2 = self.layer2(x1)  # 16x16
        x3 = self.layer3(x2)  # 8x8

        # Multi-Scale Aggregation
        feat2 = self.gap(x2).view(x2.size(0), -1)  # 128
        feat3 = self.gap(x3).view(x3.size(0), -1)  # 256

        combined = torch.cat([feat2, feat3], dim=1)
        logits = self.fc(combined)
        return logits


# ==========================================
# Training & Evaluation Logic
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        all_preds.extend(torch.sigmoid(outputs).detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_auc = roc_auc_score(all_targets, all_preds)
    return epoch_loss, epoch_auc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device).unsqueeze(1)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    val_loss = running_loss / len(loader.dataset)
    val_auc = roc_auc_score(all_targets, all_preds)
    return val_loss, val_auc


def predict_tta(model, images, device):
    """
    Test Time Augmentation: Original, H-Flip, V-Flip
    """
    model.eval()
    preds = []

    # We process one by one or batch? Batch is better.
    # But for simplicity in this function, let's assume images is a tensor batch
    # Actually, the caller will likely pass a dataloader.
    pass


def run_experiment(load_cached_data=True):
    # 1. Load Data
    data = load_data(load_cached_data=load_cached_data)

    # 2. Prepare Test Loader (Common for all seeds)
    # TTA logic will be applied during inference loop
    test_imgs_np = data["test_images"]
    test_ids = data["test_ids"]

    # 3. Training Loop per Seed
    final_test_preds = np.zeros(len(test_ids))

    for seed in SEEDS:
        print(f"\n--- Training Seed {seed} ---")
        set_seed(seed)

        # Augmentations for training
        def train_transform(img):
            # img is HWC numpy uint8
            # Random H-Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 1)
            # Random V-Flip
            if random.random() > 0.5:
                img = cv2.flip(img, 0)

            img = torch.from_numpy(img.copy().transpose((2, 0, 1))).float() / 255.0
            return img

        def eval_transform(img):
            img = torch.from_numpy(img.copy().transpose((2, 0, 1))).float() / 255.0
            return img

        # Datasets
        train_ds = CactusDataset(
            data["train_images"], data["train_labels"], transform=train_transform
        )
        val_ds = CactusDataset(
            data["val_images"], data["val_labels"], transform=eval_transform
        )

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
        )
        val_loader = DataLoader(
            val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
        )

        # Model setup
        model = CustomWideSEResNeXt(channels=CHANNELS, cardinality=CARDINALITY).to(
            DEVICE
        )
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
        criterion = nn.BCEWithLogitsLoss()

        best_auc = 0.0
        patience = 5
        no_improve = 0
        best_model_state = None

        for epoch in range(EPOCHS):
            train_loss, train_auc = train_one_epoch(
                model, train_loader, optimizer, criterion, DEVICE
            )
            val_loss, val_auc = validate(model, val_loader, criterion, DEVICE)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} AUC: {train_auc} | Val Loss: {val_loss:.4f} AUC: {val_auc}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                best_model_state = model.state_dict()
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                print("Early stopping triggered.")
                break

        # Load best model for this seed
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

        # Inference with TTA
        print(f"Generating predictions for Seed {seed}...")
        model.eval()
        seed_preds = []

        # Process test set in batches
        # We manually batch the numpy array to apply TTA
        num_test = len(test_imgs_np)
        for i in range(0, num_test, BATCH_SIZE):
            batch_imgs = test_imgs_np[i : i + BATCH_SIZE]

            # Prepare TTA versions
            # 1. Original
            t1 = torch.stack([eval_transform(img) for img in batch_imgs]).to(DEVICE)
            # 2. H-Flip
            t2 = torch.stack(
                [eval_transform(cv2.flip(img, 1)) for img in batch_imgs]
            ).to(DEVICE)
            # 3. V-Flip
            t3 = torch.stack(
                [eval_transform(cv2.flip(img, 0)) for img in batch_imgs]
            ).to(DEVICE)

            with torch.no_grad():
                p1 = torch.sigmoid(model(t1))
                p2 = torch.sigmoid(model(t2))
                p3 = torch.sigmoid(model(t3))

            # Average TTA
            avg_p = (p1 + p2 + p3) / 3.0
            seed_preds.extend(avg_p.cpu().numpy().flatten())

        final_test_preds += np.array(seed_preds)

    # Average across seeds
    final_test_preds /= len(SEEDS)

    # Save Submission
    sub_df = pd.DataFrame({"id": test_ids, "has_cactus": final_test_preds})

    save_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    sub_df.to_csv(save_path, index=False)
    print(f"Submission saved to {save_path}")

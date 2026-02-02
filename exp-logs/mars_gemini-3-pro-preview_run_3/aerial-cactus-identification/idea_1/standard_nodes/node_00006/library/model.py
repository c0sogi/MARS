import os
import cv2
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from library.utils import seed_everything, get_device

# ==========================================
# Data Processing & Caching
# ==========================================


def load_and_preprocess_data(
    metadata_path, input_dir, cache_name, load_cached_data=True
):
    """
    Loads images based on metadata, converts to arrays, and handles caching.
    Returns:
        ids (numpy array): Array of image IDs.
        images (numpy array): Array of image data (N, 32, 32, 3).
        labels (numpy array): Array of labels (if available, else zeros).
    """
    cache_dir = "./working/idea_1"
    os.makedirs(cache_dir, exist_ok=True)

    cache_path_ids = os.path.join(cache_dir, f"{cache_name}_ids.npy")
    cache_path_imgs = os.path.join(cache_dir, f"{cache_name}_imgs.npy")
    cache_path_lbls = os.path.join(cache_dir, f"{cache_name}_lbls.npy")

    # 1. Try to load from cache
    if load_cached_data:
        if (
            os.path.exists(cache_path_ids)
            and os.path.exists(cache_path_imgs)
            and os.path.exists(cache_path_lbls)
        ):
            print(f"Loading {cache_name} data from cache...")
            ids = np.load(cache_path_ids, allow_pickle=True)
            images = np.load(cache_path_imgs)
            labels = np.load(cache_path_lbls)
            return ids, images, labels

    # 2. Process from scratch
    print(f"Processing {cache_name} data from scratch...")
    df = pd.read_csv(metadata_path)

    ids = []
    images = []
    labels = []

    for _, row in df.iterrows():
        img_id = row["id"]
        rel_path = row["file_path"]
        label = row["has_cactus"]

        full_path = os.path.join(input_dir, rel_path)
        img = cv2.imread(full_path)

        if img is None:
            # Fallback for missing images (though metadata check passed)
            # Create a black image to prevent crash
            img = np.zeros((32, 32, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        ids.append(img_id)
        images.append(img)
        labels.append(label)

    ids = np.array(ids)
    images = np.array(images, dtype=np.uint8)  # Keep as uint8 to save space in cache
    labels = np.array(labels, dtype=np.float32)

    # 3. Save to cache
    np.save(cache_path_ids, ids)
    np.save(cache_path_imgs, images)
    np.save(cache_path_lbls, labels)

    return ids, images, labels


class CactusDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Convert to float [0, 1]
        img = self.images[idx].astype(np.float32) / 255.0
        label = self.labels[idx]

        # Apply augmentations if any (manual implementation for simplicity/speed)
        if self.transform:
            img = self.transform(img)

        # To Tensor (H, W, C) -> (C, H, W)
        img = torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)

        return img, label


def train_augment(img):
    # Random Horizontal Flip
    if np.random.rand() > 0.5:
        img = np.fliplr(img).copy()  # copy to avoid negative stride issues
    return img


# ==========================================
# Model Architecture
# ==========================================


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 2
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Block 3
        self.block3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classification Head
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x


# ==========================================
# Training & Evaluation Logic
# ==========================================


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device).unsqueeze(1)

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
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    try:
        epoch_auc = roc_auc_score(all_targets, all_preds)
    except ValueError:
        epoch_auc = 0.5  # Handle edge case if only one class present in batch/split

    return epoch_loss, epoch_auc


def predict(model, loader, device):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy().flatten())

    return np.array(all_preds)


# ==========================================
# Main Execution
# ==========================================


def run_cactus_classification(
    epochs=20, batch_size=64, learning_rate=1e-3, patience=5, load_cached_data=True
):
    seed_everything(42)
    device = get_device()
    print(f"Using device: {device}")

    # Paths
    input_dir = "./input"
    meta_dir = "./metadata"

    # 1. Load Data
    train_ids, train_imgs, train_lbls = load_and_preprocess_data(
        os.path.join(meta_dir, "train_metadata.csv"),
        input_dir,
        "train",
        load_cached_data,
    )
    val_ids, val_imgs, val_lbls = load_and_preprocess_data(
        os.path.join(meta_dir, "val_metadata.csv"), input_dir, "val", load_cached_data
    )
    test_ids, test_imgs, test_lbls = load_and_preprocess_data(
        os.path.join(meta_dir, "test_metadata.csv"), input_dir, "test", load_cached_data
    )

    # 2. Create Datasets & Loaders
    train_dataset = CactusDataset(train_imgs, train_lbls, transform=train_augment)
    val_dataset = CactusDataset(val_imgs, val_lbls, transform=None)
    test_dataset = CactusDataset(
        test_imgs, test_lbls, transform=None
    )  # Labels ignored for test

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # 3. Initialize Model
    model = SimpleCNN().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 4. Training Loop with Early Stopping
    best_auc = 0.0
    patience_counter = 0
    best_model_state = None

    print("\nStarting training...")
    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_auc = validate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break

    # 5. Restore Best Model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model with Val AUC: {best_auc}")

    # 6. Generate Submission
    print("Generating predictions for test set...")
    test_preds = predict(model, test_loader, device)

    submission_dir = "./submission"
    os.makedirs(submission_dir, exist_ok=True)
    submission_path = os.path.join(submission_dir, "submission.csv")

    submission_df = pd.DataFrame({"id": test_ids, "has_cactus": test_preds})

    submission_df.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    run_cactus_classification()

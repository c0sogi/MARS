import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from tqdm import tqdm

from library.config import (
    TRAIN_BSON_PATH,
    TEST_BSON_PATH,
    TRAIN_META_PATH,
    VAL_META_PATH,
    TEST_META_PATH,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    CLASS_WEIGHTS_PATH,
    MODEL_SAVE_PATH,
    SUBMISSION_FILE_PATH,
    IMG_SIZE,
    IMG_MEAN,
    IMG_STD,
    BATCH_SIZE,
    NUM_WORKERS,
    DEVICE,
    FEATURE_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    SEED,
)
from library.utils import (
    read_bson_images_at_offset,
    calculate_accuracy,
    save_submission,
)

# Ensure reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ==========================================
# MODEL ARCHITECTURES
# ==========================================


class ResNetBackbone(nn.Module):
    """
    Frozen ResNet-50 Feature Extractor.
    Outputs a 2048-dim vector per image.
    """

    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # Load pre-trained weights
        weights = models.ResNet50_Weights.DEFAULT
        self.backbone = models.resnet50(weights=weights)

        # Replace FC layer with Identity to get features
        self.backbone.fc = nn.Identity()

        # Freeze parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


class AttentionClassifier(nn.Module):
    """
    Attention-based aggregation model.
    Takes a bag of image features, computes attention weights, aggregates, and classifies.
    """

    def __init__(
        self,
        input_dim=FEATURE_DIM,
        hidden_dim=HIDDEN_DIM,
        num_classes=NUM_CLASSES,
        dropout_rate=DROPOUT_RATE,
    ):
        super(AttentionClassifier, self).__init__()

        # Attention Mechanism (Gated Attention)
        self.attention_V = nn.Linear(input_dim, hidden_dim)
        self.attention_U = nn.Linear(hidden_dim, 1)

        # Classifier MLP
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: [Batch, Max_Images, Input_Dim]
            mask: [Batch, Max_Images] (1.0 for valid, 0.0 for padding)
        """
        # 1. Compute Attention Scores
        # [B, N, D] -> [B, N, H]
        attn_hidden = torch.tanh(self.attention_V(x))
        # [B, N, H] -> [B, N, 1]
        attn_scores = self.attention_U(attn_hidden)

        # 2. Masking
        if mask is not None:
            # Expand mask to [B, N, 1]
            mask_expanded = mask.unsqueeze(-1)
            # Set scores of padded positions to very small value
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e9)

        # 3. Softmax
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, N, 1]

        # 4. Weighted Sum Aggregation
        # [B, N, 1] * [B, N, D] -> [B, N, D] -> Sum over N -> [B, D]
        weighted_features = (x * attn_weights).sum(dim=1)

        # 5. Classification
        logits = self.classifier(weighted_features)
        return logits


# ==========================================
# DATA PROCESSING & CACHING
# ==========================================


class RawImageDataset(Dataset):
    """
    Dataset to read raw images from BSON for feature extraction.
    Handles file opening per-worker to avoid pickling issues.
    """

    def __init__(self, metadata_path, bson_path, transform=None):
        self.meta = pd.read_csv(metadata_path)
        self.bson_path = bson_path
        self.transform = transform
        self.bson_file = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        # Lazy file opening
        if self.bson_file is None:
            self.bson_file = open(self.bson_path, "rb")

        row = self.meta.iloc[idx]
        offset = row["bson_offset"]
        length = row["bson_length"]

        # Read images (returns list of BGR numpy arrays)
        images_np = read_bson_images_at_offset(self.bson_file, offset, length)

        if len(images_np) == 0:
            # Fallback for empty records (should not happen in clean data)
            images_np = [np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)]

        processed_imgs = []
        if self.transform:
            for img in images_np:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_imgs.append(self.transform(img_rgb))

        # Stack: [N, C, H, W]
        if len(processed_imgs) > 0:
            img_tensor = torch.stack(processed_imgs)
        else:
            img_tensor = torch.zeros((1, 3, IMG_SIZE, IMG_SIZE))

        return img_tensor, idx

    def __del__(self):
        if self.bson_file:
            self.bson_file.close()


def get_transforms():
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=IMG_MEAN, std=IMG_STD),
        ]
    )


def collate_raw_images(batch):
    """
    Flattens a batch of image lists into a single large batch for ResNet.
    """
    flat_images = []
    counts = []
    meta_indices = []

    for imgs, idx in batch:
        flat_images.append(imgs)
        counts.append(imgs.shape[0])
        meta_indices.append(idx)

    flat_images = torch.cat(flat_images, dim=0)
    return flat_images, counts, meta_indices


def extract_features(
    metadata_path,
    bson_path,
    output_feat_path,
    output_label_path=None,
    output_id_path=None,
    load_cached_data=True,
):
    """
    Extracts features using ResNetBackbone and caches them to disk.
    """
    # Check if cache exists
    base_dir = os.path.dirname(output_feat_path)
    os.makedirs(base_dir, exist_ok=True)

    basename = os.path.basename(output_feat_path).replace(".npy", "")
    index_path = os.path.join(base_dir, f"{basename}_index.npy")

    cache_exists = (
        os.path.exists(output_feat_path)
        and os.path.exists(index_path)
        and (output_label_path is None or os.path.exists(output_label_path))
        and (output_id_path is None or os.path.exists(output_id_path))
    )

    if load_cached_data and cache_exists:
        print(f"Loading cached features from {output_feat_path}...")
        flat_features = np.load(output_feat_path)
        indices = np.load(index_path)
        labels = np.load(output_label_path) if output_label_path else None
        ids = np.load(output_id_path) if output_id_path else None
        return flat_features, indices, labels, ids

    print(f"Extracting features from {bson_path}...")

    dataset = RawImageDataset(metadata_path, bson_path, transform=get_transforms())
    # Use a reasonable batch size for inference
    dataloader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_raw_images,
        pin_memory=True,
    )

    model = ResNetBackbone().to(DEVICE)
    model.eval()

    all_features = []
    all_counts = []

    with torch.no_grad():
        for imgs, counts, _ in tqdm(dataloader, desc="Feature Extraction"):
            imgs = imgs.to(DEVICE)
            feats = model(imgs)  # [Sum_N, 2048]
            feats = feats.cpu().numpy().astype(np.float32)
            all_features.append(feats)
            all_counts.extend(counts)

    # Combine
    flat_features = np.concatenate(all_features, axis=0)

    # Build Index [Start, Count]
    counts_arr = np.array(all_counts, dtype=np.int32)
    starts_arr = np.concatenate(([0], np.cumsum(counts_arr)[:-1])).astype(np.int32)
    indices = np.stack([starts_arr, counts_arr], axis=1)

    # Save
    np.save(output_feat_path, flat_features)
    np.save(index_path, indices)

    df = pd.read_csv(metadata_path)
    labels = None
    ids = None

    if output_label_path and "category_id" in df.columns:
        labels = df["category_id"].values.astype(np.int64)
        np.save(output_label_path, labels)

    if output_id_path:
        ids = df["_id"].values.astype(np.int64)
        np.save(output_id_path, ids)

    return flat_features, indices, labels, ids


# ==========================================
# TRAINING & INFERENCE
# ==========================================


class FeatureDataset(Dataset):
    """
    Dataset for training the classifier on pre-computed features.
    """

    def __init__(self, flat_features, indices, labels=None, max_len=4):
        self.flat_features = flat_features
        self.indices = indices
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start, count = self.indices[idx]

        # Retrieve bag
        bag = self.flat_features[start : start + count]

        # Pad/Truncate
        L, D = bag.shape
        if L >= self.max_len:
            bag = bag[: self.max_len]
            mask = np.ones(self.max_len, dtype=np.float32)
        else:
            padding = np.zeros((self.max_len - L, D), dtype=bag.dtype)
            bag = np.concatenate([bag, padding], axis=0)
            mask = np.concatenate(
                [np.ones(L), np.zeros(self.max_len - L)], axis=0
            ).astype(np.float32)

        bag_tensor = torch.from_numpy(bag)
        mask_tensor = torch.from_numpy(mask)

        if self.labels is not None:
            return bag_tensor, mask_tensor, self.labels[idx]
        return bag_tensor, mask_tensor


def get_category_mapping():
    # Helper to map raw category_ids to 0..N indices
    cats = pd.read_csv(
        os.path.join(os.path.dirname(TRAIN_META_PATH), "../input/category_names.csv")
    )
    unique_cats = sorted(cats["category_id"].unique())
    cat2idx = {cat: i for i, cat in enumerate(unique_cats)}
    idx2cat = {i: cat for i, cat in enumerate(unique_cats)}
    return cat2idx, idx2cat


def train_model(load_cached_data=True):
    print("=== Starting Training Pipeline ===")

    # 1. Load Data
    train_feats, train_idx, train_labels, _ = extract_features(
        TRAIN_META_PATH,
        TRAIN_BSON_PATH,
        TRAIN_FEATURES_PATH,
        TRAIN_LABELS_PATH,
        load_cached_data=load_cached_data,
    )
    val_feats, val_idx, val_labels, _ = extract_features(
        VAL_META_PATH,
        TRAIN_BSON_PATH,
        VAL_FEATURES_PATH,
        VAL_LABELS_PATH,
        load_cached_data=load_cached_data,
    )

    # 2. Map Labels
    cat2idx, _ = get_category_mapping()
    train_y = np.array([cat2idx[l] for l in train_labels])
    val_y = np.array([cat2idx[l] for l in val_labels])

    # 3. Compute Class Weights
    if os.path.exists(CLASS_WEIGHTS_PATH) and load_cached_data:
        weights_np = np.load(CLASS_WEIGHTS_PATH)
        class_weights = torch.from_numpy(weights_np).float().to(DEVICE)
    else:
        counts = np.bincount(train_y, minlength=NUM_CLASSES)
        weights_np = 1.0 / (counts + 1.0)
        weights_np = weights_np / weights_np.mean()
        class_weights = torch.tensor(weights_np, dtype=torch.float32).to(DEVICE)
        np.save(CLASS_WEIGHTS_PATH, weights_np)

    # 4. Setup Loaders
    train_ds = FeatureDataset(train_feats, train_idx, train_y)
    val_ds = FeatureDataset(val_feats, val_idx, val_y)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # 5. Model & Opt
    model = AttentionClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        steps_per_epoch=len(train_loader),
        epochs=NUM_EPOCHS,
    )

    # 6. Loop
    best_acc = -1.0
    patience_counter = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0

        for bags, masks, targets in train_loader:
            bags, masks, targets = bags.to(DEVICE), masks.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(bags, masks)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            scheduler.step()

            train_loss += loss.item() * bags.size(0)

        train_loss /= len(train_ds)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for bags, masks, targets in val_loader:
                bags, masks, targets = (
                    bags.to(DEVICE),
                    masks.to(DEVICE),
                    targets.to(DEVICE),
                )
                outputs = model(bags, masks)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * bags.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss /= len(val_ds)
        val_acc = calculate_accuracy(all_preds, all_targets)

        print(
            f"Epoch {epoch+1}/{NUM_EPOCHS} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val Acc: {val_acc:.6f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered.")
                break

    print(f"Training Complete. Best Validation Accuracy: {best_acc:.6f}")
    return model


def predict_submission(load_cached_data=True):
    print("=== Generating Submission ===")

    # 1. Load Test Data
    test_feats, test_idx, _, test_ids = extract_features(
        TEST_META_PATH,
        TEST_BSON_PATH,
        TEST_FEATURES_PATH,
        output_id_path=TEST_IDS_PATH,
        load_cached_data=load_cached_data,
    )

    # 2. Load Model
    model = AttentionClassifier().to(DEVICE)
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    else:
        print("Error: Model file not found!")
        return

    model.eval()

    # 3. Predict
    test_ds = FeatureDataset(test_feats, test_idx, labels=None)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    all_preds_idx = []
    with torch.no_grad():
        for bags, masks in tqdm(test_loader, desc="Inference"):
            bags, masks = bags.to(DEVICE), masks.to(DEVICE)
            outputs = model(bags, masks)
            preds = torch.argmax(outputs, dim=1)
            all_preds_idx.extend(preds.cpu().numpy())

    # 4. Map to Category IDs
    _, idx2cat = get_category_mapping()
    final_preds = [idx2cat[idx] for idx in all_preds_idx]

    # 5. Save
    save_submission(test_ids, final_preds, SUBMISSION_FILE_PATH)

import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tqdm.auto import tqdm
import glob


# ====================================================
# Configuration
# ====================================================
class Config:
    # General
    seed = 42
    num_workers = 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data
    input_dir = "./input"
    metadata_dir = "./metadata"
    working_dir = "./working"
    images_dir = os.path.join(input_dir, "images")

    # Training
    n_folds = 5
    epochs = 25
    patience = 12
    batch_size = 16  # Adjusted for A100 and model sizes
    learning_rate = 2e-4
    min_lr = 1e-6
    weight_decay = 1e-4
    max_grad_norm = 10.0

    # Models
    # List of (backbone_name, image_size) tuples
    models_config = [
        ("tf_efficientnetv2_m.in21k_ft_in1k", 384),
        ("maxvit_small_tf_224.in1k", 224),
    ]

    # Target Columns (Alphabetical order as per sample submission usually)
    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    num_classes = 4


# ====================================================
# Utils
# ====================================================
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_class_weights(df, target_cols):
    # Calculate inverse frequency weights
    # Weight = Total_Samples / (Num_Classes * Class_Count)
    # We use the 'stratify_label' or max of one-hot to count
    labels = df[target_cols].idxmax(axis=1)
    counts = labels.value_counts().sort_index()
    total = len(df)
    n_classes = len(target_cols)

    weights = []
    for col in target_cols:
        count = counts.get(col, 0)
        if count > 0:
            w = total / (n_classes * count)
        else:
            w = 1.0
        weights.append(w)

    return torch.tensor(weights, dtype=torch.float32)


# ====================================================
# Dataset
# ====================================================
class AppleDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        self.df = df
        self.image_dir = image_dir
        self.transform = transform
        self.file_paths = df["file_path"].values

        # Check if targets exist (Train/Val) or not (Test)
        self.targets = None
        if set(Config.target_cols).issubset(df.columns):
            self.targets = df[Config.target_cols].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # file_path in metadata is relative, e.g., "images/Test_0.jpg"
        # but input_dir is "./input", so full path is "./input/images/Test_0.jpg"
        # The metadata generation script joined IMAGES_DIR already.
        # Let's construct the full path carefully.
        rel_path = self.file_paths[idx]
        file_path = os.path.join(Config.input_dir, rel_path)

        image = cv2.imread(file_path)
        if image is None:
            # Fallback or error handling
            raise FileNotFoundError(f"Image not found at {file_path}")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        if self.targets is not None:
            label = torch.tensor(self.targets[idx], dtype=torch.float32)
            # For CrossEntropy, we usually need indices if strictly single label,
            # but here we might have soft labels or we use argmax later.
            # The task implies mutually exclusive mostly, but "multiple_diseases" is a class.
            # We will use CrossEntropy, so we need the class index.
            label_idx = torch.argmax(label)
            return image, label_idx
        else:
            return image, torch.tensor(0)  # Dummy label for test


def get_transforms(data_type, img_size):
    if data_type == "train":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                # Strong Geometric Augmentations
                A.ShiftScaleRotate(
                    shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.5
                ),
                # No Cutout, No Brightness/Contrast as per Idea
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data_type == "valid" or data_type == "test":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )
    elif data_type == "tta_flip":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=1.0),  # Force flip
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
                ToTensorV2(),
            ]
        )


# ====================================================
# Model
# ====================================================
class GeM(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(
            1.0 / p
        )


class AppleDiseaseModel(nn.Module):
    def __init__(self, backbone_name, num_classes, pretrained=True):
        super(AppleDiseaseModel, self).__init__()
        # Use features_only to get intermediate layers
        # We want the last 3 stages. Indices depend on the model, but usually 2,3,4 work for 5-stage models.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )

        feature_info = self.backbone.feature_info.channels()
        self.gem_pools = nn.ModuleList([GeM() for _ in range(len(feature_info))])

        # Calculate total input features for the linear layer
        total_features = sum(feature_info)
        self.fc = nn.Linear(total_features, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        # Apply GeM pooling to each extracted feature map and flatten
        pooled_features = [
            gem(f).flatten(1) for gem, f in zip(self.gem_pools, features)
        ]
        # Concatenate
        concat_features = torch.cat(pooled_features, dim=1)
        output = self.fc(concat_features)
        return output


# ====================================================
# Training Helper Functions
# ====================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), Config.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds = []
    targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            # Apply softmax for AUC calculation
            probs = torch.softmax(outputs, dim=1)
            preds.append(probs.cpu().numpy())
            targets.append(labels.cpu().numpy())

    preds = np.concatenate(preds)
    targets = np.concatenate(targets)

    # One-hot encode targets for ROC AUC
    # targets is (N,), preds is (N, C)
    # We need to one-hot targets
    targets_one_hot = np.zeros_like(preds)
    targets_one_hot[np.arange(len(targets)), targets] = 1

    try:
        auc = roc_auc_score(targets_one_hot, preds, average="macro", multi_class="ovr")
    except ValueError:
        auc = 0.5  # Fallback if single class in batch

    return running_loss / len(loader.dataset), auc


# ====================================================
# Main Pipeline
# ====================================================
def run_training():
    seed_everything(Config.seed)
    os.makedirs(Config.working_dir, exist_ok=True)

    # Load Metadata
    train_full = pd.read_csv(os.path.join(Config.metadata_dir, "train.csv"))
    val_full = pd.read_csv(os.path.join(Config.metadata_dir, "val.csv"))

    # Combine for cross-validation loop (since we want to control folds ourselves or use provided splits)
    # The provided metadata has a fixed train/val split.
    # To perform 5-fold CV as requested in the Idea, we should merge and create folds,
    # OR if the metadata implies a fixed split, we might just train on train and val on val.
    # However, the Idea explicitly says "5-Fold Stratified Cross-Validation".
    # So we will merge and stratified K-fold.
    full_df = pd.concat([train_full, val_full]).reset_index(drop=True)

    # Create Stratified Folds
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    # We need a label for stratification. 'stratify_label' exists in metadata.
    full_df["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(full_df, full_df["stratify_label"])):
        full_df.loc[val_idx, "fold"] = fold

    # Calculate Class Weights based on full dataset
    class_weights = get_class_weights(full_df, Config.target_cols).to(Config.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Loop over Backbones
    for backbone_name, img_size in Config.models_config:
        print(f"\nTraining Backbone: {backbone_name} (Size: {img_size})")

        # Loop over Folds
        for fold in range(Config.n_folds):
            print(f"  Fold {fold+1}/{Config.n_folds}")

            train_df = full_df[full_df["fold"] != fold].reset_index(drop=True)
            valid_df = full_df[full_df["fold"] == fold].reset_index(drop=True)

            train_dataset = AppleDataset(
                train_df, Config.images_dir, transform=get_transforms("train", img_size)
            )
            valid_dataset = AppleDataset(
                valid_df, Config.images_dir, transform=get_transforms("valid", img_size)
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=Config.batch_size,
                shuffle=True,
                num_workers=Config.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            valid_loader = DataLoader(
                valid_dataset,
                batch_size=Config.batch_size,
                shuffle=False,
                num_workers=Config.num_workers,
                pin_memory=True,
            )

            model = AppleDiseaseModel(backbone_name, Config.num_classes).to(
                Config.device
            )

            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=Config.learning_rate,
                weight_decay=Config.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.epochs, eta_min=Config.min_lr
            )
            scaler = GradScaler()

            best_auc = 0.0
            patience_counter = 0

            model_save_path = os.path.join(
                Config.working_dir, f"{backbone_name.replace('.', '_')}_fold_{fold}.pth"
            )

            for epoch in range(Config.epochs):
                train_loss = train_one_epoch(
                    model, train_loader, criterion, optimizer, scaler, Config.device
                )
                val_loss, val_auc = validate(
                    model, valid_loader, criterion, Config.device
                )

                scheduler.step()

                print(
                    f"    Epoch {epoch+1}: Train Loss {train_loss:.4f}, Val Loss {val_loss:.4f}, Val AUC {val_auc:.6f}"
                )

                if val_auc > best_auc:
                    best_auc = val_auc
                    torch.save(model.state_dict(), model_save_path)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= Config.patience:
                    print(f"    Early stopping at epoch {epoch+1}")
                    break

            # Clean up to save memory
            del model, optimizer, scaler, scheduler
            torch.cuda.empty_cache()


def run_inference():
    print("\nStarting Inference...")
    test_df = pd.read_csv(os.path.join(Config.metadata_dir, "test.csv"))

    # We will accumulate predictions
    final_preds = np.zeros((len(test_df), Config.num_classes))
    model_count = 0

    for backbone_name, img_size in Config.models_config:
        # Prepare Datasets for TTA
        # 1. Original
        ds_orig = AppleDataset(
            test_df, Config.images_dir, transform=get_transforms("valid", img_size)
        )
        loader_orig = DataLoader(
            ds_orig,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        # 2. Horizontal Flip
        ds_flip = AppleDataset(
            test_df, Config.images_dir, transform=get_transforms("tta_flip", img_size)
        )
        loader_flip = DataLoader(
            ds_flip,
            batch_size=Config.batch_size,
            shuffle=False,
            num_workers=Config.num_workers,
        )

        for fold in range(Config.n_folds):
            model_path = os.path.join(
                Config.working_dir, f"{backbone_name.replace('.', '_')}_fold_{fold}.pth"
            )
            if not os.path.exists(model_path):
                print(f"Warning: Model {model_path} not found. Skipping.")
                continue

            model = AppleDiseaseModel(
                backbone_name, Config.num_classes, pretrained=False
            )
            model.load_state_dict(torch.load(model_path, map_location=Config.device))
            model.to(Config.device)
            model.eval()

            # Predict Original
            preds_orig = []
            with torch.no_grad():
                for images, _ in loader_orig:
                    images = images.to(Config.device)
                    outputs = model(images)
                    preds_orig.append(torch.softmax(outputs, dim=1).cpu().numpy())
            preds_orig = np.concatenate(preds_orig)

            # Predict Flip
            preds_flip = []
            with torch.no_grad():
                for images, _ in loader_flip:
                    images = images.to(Config.device)
                    outputs = model(images)
                    preds_flip.append(torch.softmax(outputs, dim=1).cpu().numpy())
            preds_flip = np.concatenate(preds_flip)

            # Average TTA
            fold_preds = (preds_orig + preds_flip) / 2.0

            final_preds += fold_preds
            model_count += 1

            del model
            torch.cuda.empty_cache()

    if model_count > 0:
        final_preds /= model_count

    # Create Submission DataFrame
    submission = pd.DataFrame(final_preds, columns=Config.target_cols)
    submission.insert(0, "image_id", test_df["image_id"])

    # Save
    submission_path = os.path.join("submission", "submission.csv")
    os.makedirs("submission", exist_ok=True)
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")


def run_task():
    run_training()
    run_inference()

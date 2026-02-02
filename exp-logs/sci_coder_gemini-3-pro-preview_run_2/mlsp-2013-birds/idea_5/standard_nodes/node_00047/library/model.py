import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    densenet121,
    DenseNet121_Weights,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from library.utils import set_seed, calculate_pos_weights, mixup_data, mixup_criterion

# ==========================================
# Configuration
# ==========================================
CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "epochs": 25,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "num_folds": 5,
    "mixup_alpha": 0.4,
    "image_size": 224,
    "num_classes": 19,
    "num_tiles": 3,
    "input_dir": "./input",
    "metadata_dir": "./metadata",
    "working_dir": "./working/idea_5",
    "submission_dir": "./submission",
}


# ==========================================
# Dataset Class
# ==========================================
class BirdDataset(Dataset):
    def __init__(self, df, root_dir, transform=None, train=True):
        self.df = df
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.labels = [c for c in df.columns if c.startswith("species_")]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Construct path to filtered spectrograms
        # Metadata has "supplemental_data/spectrograms/...", we want "supplemental_data/filtered_spectrograms/..."
        rel_path = row["file_path_spec"]
        rel_path = rel_path.replace("spectrograms", "filtered_spectrograms")
        img_path = os.path.join(self.root_dir, rel_path)

        # Load Image
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            # Fallback to blank image if file missing (though metadata check passed)
            image = np.zeros((256, 1000), dtype=np.uint8)

        # Tiling Logic
        # Image shape is (H, W). We want 3 tiles along W.
        h, w = image.shape
        tiles = []

        # Define tile coordinates (overlapping)
        # We want to cover the whole width with 3 tiles.
        # Simple strategy: Start, Center, End.
        # Tile width: let's take a chunk that preserves aspect ratio reasonably when resized to 224x224.
        # But to ensure coverage, let's just take 3 overlapping crops.

        crop_width = int(w * 0.45)  # 3 * 0.45 = 1.35 coverage (overlap)

        # Tile 1: Start
        t1 = image[:, 0:crop_width]
        # Tile 2: Center
        center_start = (w - crop_width) // 2
        t2 = image[:, center_start : center_start + crop_width]
        # Tile 3: End
        t3 = image[:, w - crop_width :]

        raw_tiles = [t1, t2, t3]
        processed_tiles = []

        for t in raw_tiles:
            # Resize to target size
            t_resized = cv2.resize(t, (CONFIG["image_size"], CONFIG["image_size"]))

            # Normalize to 0-1
            t_norm = t_resized.astype(np.float32) / 255.0

            # Add channel dimension -> (1, H, W)
            t_tensor = torch.from_numpy(t_norm).unsqueeze(0)
            processed_tiles.append(t_tensor)

        # Stack tiles -> (3, 1, H, W)
        # Note: The model expects (3, C, H, W). Here C=1.
        input_tensor = torch.stack(processed_tiles, dim=0)

        if self.train:
            label_vec = row[self.labels].values.astype(np.float32)
            return input_tensor, torch.tensor(label_vec)
        else:
            # For test, return ID as well for tracking if needed, or just dummy labels
            return input_tensor, torch.tensor(
                np.zeros(len(self.labels)), dtype=torch.float32
            )


# ==========================================
# Model Architecture
# ==========================================
class MILResNet18(nn.Module):
    def __init__(self, num_classes=19, pretrained=True):
        super(MILResNet18, self).__init__()

        # Load backbone
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # Modify first layer for 1-channel input
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # We sum the weights across the channel dimension to preserve pretrained features
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            # Sum weights across input channels (dim 1) -> (64, 1, 7, 7)
            new_conv.weight.copy_(torch.sum(old_conv.weight, dim=1, keepdim=True))

        self.backbone.conv1 = new_conv

        # Replace FC layer
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        # x shape: (Batch, Tiles, Channels, Height, Width) -> (B, 3, 1, 224, 224)
        b, t, c, h, w = x.size()

        # Merge Batch and Tiles dimensions
        x = x.view(b * t, c, h, w)

        # Pass through backbone
        logits = self.backbone(x)  # (B*T, Num_Classes)

        # Reshape back to separate tiles
        logits = logits.view(b, t, -1)  # (B, 3, Num_Classes)

        # MIL Aggregation: Max Pooling across tiles
        # We take the max logit for each class across the 3 tiles
        aggregated_logits, _ = torch.max(logits, dim=1)  # (B, Num_Classes)

        return aggregated_logits


# ==========================================
# Training & Evaluation Logic
# ==========================================
def train_one_epoch(model, loader, criterion, optimizer, device, alpha):
    model.train()
    running_loss = 0.0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Mixup
        inputs_reshaped = inputs.view(
            -1, *inputs.shape[2:]
        )  # Flatten B and T for mixup?
        # No, mixup should happen at the Bag level (Batch level).
        # inputs: (B, 3, 1, H, W)

        mixed_inputs, y_a, y_b, lam = mixup_data(inputs, targets, alpha, device)

        optimizer.zero_grad()
        outputs = model(mixed_inputs)
        loss = mixup_criterion(criterion, outputs, y_a, y_b, lam)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # Calculate ROC AUC (Macro Average)
    try:
        auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        auc = 0.5  # Handle edge cases with single class present

    return running_loss / len(loader.dataset), auc, all_preds


def run_training_and_submission():
    set_seed(CONFIG["seed"])
    os.makedirs(CONFIG["working_dir"], exist_ok=True)
    os.makedirs(CONFIG["submission_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Metadata
    train_df = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "train.csv"))
    test_df = pd.read_csv(os.path.join(CONFIG["metadata_dir"], "test.csv"))

    # Prepare Labels for Stratification
    # Since it's multi-label, we can stratify based on the combination of labels or primary label.
    # For simplicity and robustness, we use the 'fold' logic if provided, or simple KFold.
    # The prompt suggests implementing Stratified K-Fold.
    # We will use the most frequent label for stratification proxy.
    label_cols = [c for c in train_df.columns if c.startswith("species_")]
    y_labels = train_df[label_cols].values

    # Proxy for stratification: Convert multi-label to string representation
    y_str = ["".join(map(str, row.astype(int))) for row in y_labels]

    skf = StratifiedKFold(
        n_splits=CONFIG["num_folds"], shuffle=True, random_state=CONFIG["seed"]
    )

    # Store test predictions from each fold
    test_preds_sum = np.zeros((len(test_df), CONFIG["num_classes"]))

    # K-Fold Loop
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_df, y_str)):
        print(f"\n=== Fold {fold} ===")

        fold_train_df = train_df.iloc[train_idx].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_idx].reset_index(drop=True)

        # Datasets
        train_ds = BirdDataset(fold_train_df, CONFIG["input_dir"], train=True)
        val_ds = BirdDataset(fold_val_df, CONFIG["input_dir"], train=True)

        train_loader = DataLoader(
            train_ds,
            batch_size=CONFIG["batch_size"],
            shuffle=True,
            num_workers=2,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
        )

        # Model
        model = MILResNet18(num_classes=CONFIG["num_classes"], pretrained=True).to(
            device
        )

        # Loss & Optimizer
        # Calculate pos_weights for this fold
        y_train_tensor = torch.tensor(fold_train_df[label_cols].values)
        pos_weights = calculate_pos_weights(y_train_tensor, device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

        optimizer = optim.AdamW(
            model.parameters(), lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"]
        )

        best_auc = 0.0
        best_model_path = os.path.join(CONFIG["working_dir"], f"model_fold_{fold}.pth")

        for epoch in range(CONFIG["epochs"]):
            train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, device, CONFIG["mixup_alpha"]
            )
            val_loss, val_auc, _ = validate(model, val_loader, criterion, device)
            scheduler.step()

            print(
                f"Epoch {epoch+1}/{CONFIG['epochs']} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Val AUC: {val_auc:.5f}"
            )

            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        print(f"Fold {fold} Best AUC: {best_auc:.5f}")

        # Inference on Test Set with Best Model
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()

        test_ds = BirdDataset(test_df, CONFIG["input_dir"], train=False)
        test_loader = DataLoader(
            test_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=2
        )

        fold_preds = []
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                probs = torch.sigmoid(outputs)
                fold_preds.append(probs.cpu().numpy())

        fold_preds = np.vstack(fold_preds)
        test_preds_sum += fold_preds

    # Average Predictions
    avg_preds = test_preds_sum / CONFIG["num_folds"]

    # Create Submission
    submission_rows = []
    rec_ids = test_df["rec_id"].values

    for i, rec_id in enumerate(rec_ids):
        probs = avg_preds[i]
        for species_idx, prob in enumerate(probs):
            # ID format: rec_id * 100 + species_id
            row_id = rec_id * 100 + species_idx
            submission_rows.append({"Id": int(row_id), "Probability": prob})

    sub_df = pd.DataFrame(submission_rows)
    sub_path = os.path.join(CONFIG["submission_dir"], "submission.csv")
    sub_df.to_csv(sub_path, index=False)
    print(f"Submission saved to {sub_path}")


# Note: The function run_training_and_submission() is intended to be called by an external script
# or can be executed if this module is run directly (though __main__ block is omitted as per instructions).

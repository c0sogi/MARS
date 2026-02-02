import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import seed_everything, get_weighted_log_loss
from library.dataset import CervicalSpineDataset
from library.model import AnatomicallyGuidedResNet


# ==========================================
# Loss Function
# ==========================================
class HierarchicalCompoundLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Reduction='mean' averages the loss over the batch (and classes if not split)
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, logits, targets):
        """
        logits: (Batch, 8) -> [C1...C7, patient_overall]
        targets: (Batch, 8)
        """
        # Vertebral Loss: Average BCE over C1-C7
        # shape: (Batch, 7)
        loss_vert = self.bce(logits[:, :7], targets[:, :7])

        # Patient Loss: Average BCE over patient_overall
        # shape: (Batch, 1)
        loss_patient = self.bce(logits[:, 7], targets[:, 7])

        # Summing them creates the implicit weighting
        # L_vert is scaled by 1/7 relative to L_patient per unit of error
        return loss_vert + loss_patient


# ==========================================
# Training & Validation Steps
# ==========================================
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for batch_idx, (images, positions, targets, _) in enumerate(loader):
        images = images.to(device)
        positions = positions.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        # Forward pass
        logits = model(images, positions)

        # Compute loss
        loss = criterion(logits, targets)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, criterion, device, df_meta):
    model.eval()
    total_loss = 0.0

    all_preds = []
    all_targets = []
    all_study_ids = []

    with torch.no_grad():
        for images, positions, targets, study_ids in loader:
            images = images.to(device)
            positions = positions.to(device)
            targets = targets.to(device)

            logits = model(images, positions)
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Apply sigmoid for probabilities
            probs = torch.sigmoid(logits)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_study_ids.extend(study_ids)

    avg_loss = total_loss / len(loader)

    # Concatenate results
    all_preds = np.concatenate(all_preds, axis=0)  # (N, 8)
    all_targets = np.concatenate(all_targets, axis=0)  # (N, 8)

    # Reconstruct DataFrame for Metric Calculation
    # We need to flatten the (N, 8) predictions into rows like sample_submission.csv
    # Columns: C1, C2, C3, C4, C5, C6, C7, patient_overall
    class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    row_ids = []
    fractured_true = []
    fractured_pred = []

    for i, study_id in enumerate(all_study_ids):
        for j, class_name in enumerate(class_names):
            row_id = f"{study_id}_{class_name}"
            row_ids.append(row_id)
            fractured_true.append(all_targets[i, j])
            fractured_pred.append(all_preds[i, j])

    solution_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_true})

    submission_df = pd.DataFrame({"row_id": row_ids, "fractured": fractured_pred})

    # Calculate Metric
    metric = get_weighted_log_loss(solution_df, submission_df)

    return avg_loss, metric


# ==========================================
# Main Training Routine
# ==========================================
def run_training(debug=Config.DEBUG, load_cached_data=True):
    seed_everything(Config.SEED)

    # 1. Load Metadata
    train_df = pd.read_csv(Config.TRAIN_METADATA_PATH)
    val_df = pd.read_csv(Config.VAL_METADATA_PATH)

    if debug:
        train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
        val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)

    # 2. Define Transforms
    train_transforms = A.Compose(
        [
            A.Rotate(limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    val_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    # 3. Datasets & Loaders
    train_dataset = CervicalSpineDataset(
        train_df,
        mode="train",
        transforms=train_transforms,
        load_cached_data=load_cached_data,
    )
    val_dataset = CervicalSpineDataset(
        val_df, mode="val", transforms=val_transforms, load_cached_data=load_cached_data
    )

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

    # 4. Model, Optimizer, Scheduler, Loss
    device = torch.device(Config.DEVICE)
    model = AnatomicallyGuidedResNet(pretrained=Config.PRETRAINED).to(device)

    criterion = HierarchicalCompoundLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # T_max = EPOCHS * 1.5
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(Config.EPOCHS * Config.T_MAX_MULTIPLIER)
    )

    # 5. Training Loop
    best_metric = float("inf")
    patience_counter = 0

    print(f"Starting training on {device} for {Config.EPOCHS} epochs.")

    for epoch in range(Config.EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_metric = validate(model, val_loader, criterion, device, val_df)

        scheduler.step()

        print(f"Epoch {epoch+1}/{Config.EPOCHS}")
        print(f"Train Loss: {train_loss}")
        print(f"Val Loss: {val_loss}")
        print(f"Val Metric: {val_metric}")

        # Early Stopping & Checkpointing
        if val_metric < (best_metric - Config.MIN_DELTA):
            best_metric = val_metric
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(Config.SUBMISSION_DIR, "best_model.pth"),
            )
            print("New best model saved.")
        else:
            patience_counter += 1

        if patience_counter >= Config.PATIENCE:
            print("Early stopping triggered.")
            break

    print(f"Training complete. Best Metric: {best_metric}")


# ==========================================
# Inference Routine
# ==========================================
def inference(model_path, load_cached_data=True):
    seed_everything(Config.SEED)
    device = torch.device(Config.DEVICE)

    # 1. Load Test Metadata
    test_df = pd.read_csv(Config.TEST_METADATA_PATH)

    # 2. Setup Dataset & Loader
    test_transforms = A.Compose(
        [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

    test_dataset = CervicalSpineDataset(
        test_df,
        mode="test",
        transforms=test_transforms,
        load_cached_data=load_cached_data,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Load Model
    model = AnatomicallyGuidedResNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 4. Generate Predictions
    study_preds = {}  # {study_id: {class: prob}}
    class_names = [f"C{i}" for i in range(1, 8)] + ["patient_overall"]

    with torch.no_grad():
        for images, positions, _, study_ids in test_loader:
            images = images.to(device)
            positions = positions.to(device)

            logits = model(images, positions)
            probs = torch.sigmoid(logits).cpu().numpy()

            for i, study_id in enumerate(study_ids):
                study_preds[study_id] = {}
                for j, cls in enumerate(class_names):
                    study_preds[study_id][cls] = probs[i, j]

    # 5. Format Submission
    sample_sub = pd.read_csv(Config.SAMPLE_SUBMISSION_PATH)

    # Helper to lookup prediction
    def get_pred(row_id):
        # row_id format: StudyInstanceUID_Class
        # But patient_overall has an underscore.
        if row_id.endswith("patient_overall"):
            study_id = row_id.replace("_patient_overall", "")
            cls = "patient_overall"
        else:
            parts = row_id.split("_")
            cls = parts[-1]
            study_id = "_".join(parts[:-1])

        if study_id in study_preds and cls in study_preds[study_id]:
            return study_preds[study_id][cls]
        else:
            # Fallback (should not happen if metadata is consistent)
            return 0.5

    sample_sub["fractured"] = sample_sub["row_id"].apply(get_pred)

    # 6. Save
    sample_sub.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, verify_initial_loss
from library.dataset import AppleDataset, get_transforms
from library.model import get_model
from library.loss import WeightedSoftTargetCrossEntropy


def get_data_split():
    """
    Loads train and validation metadata separately for fixed-split training.
    """
    if not os.path.exists(Config.train_metadata_path) or not os.path.exists(
        Config.val_metadata_path
    ):
        raise FileNotFoundError("Metadata files not found.")

    df_train = pd.read_csv(Config.train_metadata_path)
    df_val = pd.read_csv(Config.val_metadata_path)

    # Ensure stratify_label exists (useful for analysis even if not splitting)
    for df in [df_train, df_val]:
        if "stratify_label" not in df.columns:
            df["stratify_label"] = df[Config.target_cols].idxmax(axis=1)

    return df_train, df_val


def calculate_class_weights(df: pd.DataFrame, device):
    """
    Calculates class weights based on inverse frequency.
    """
    # Count instances per class using the stratify_label
    class_counts = df["stratify_label"].value_counts().sort_index()
    # Ensure order matches Config.target_cols
    counts = np.array([class_counts.get(cls, 0) for cls in Config.target_cols])

    total_samples = len(df)
    n_classes = len(Config.target_cols)

    # Formula: N_total / (N_classes * N_samples_per_class)
    weights = total_samples / (n_classes * counts)

    # Normalize weights so they sum to n_classes (optional, but keeps loss scale similar)
    # or just keep as is. PyTorch CrossEntropy usually takes raw weights.

    print("Class Weights:", dict(zip(Config.target_cols, weights)))

    return torch.tensor(weights, dtype=torch.float32).to(device)


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    dataset_size = 0

    for images, targets in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        batch_size = images.size(0)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_size
        dataset_size += batch_size

    epoch_loss = running_loss / dataset_size
    return epoch_loss


def validate(model, dataloader, criterion, device):
    """
    Evaluates the model on the validation set.
    """
    model.eval()
    running_loss = 0.0
    dataset_size = 0

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device)

            batch_size = images.size(0)

            outputs = model(images)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

            # Apply softmax for AUC calculation
            probs = torch.softmax(outputs, dim=1)

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    val_loss = running_loss / dataset_size

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Calculate ROC AUC
    # average='macro' calculates metrics for each label, and finds their unweighted mean.
    try:
        val_auc = roc_auc_score(all_targets, all_preds, average="macro")
    except ValueError:
        # Handle edge cases where a class might be missing in validation batch (unlikely with StratifiedKFold)
        val_auc = 0.5

    return val_loss, val_auc


def run_seed(seed, df_train, df_val, class_weights):
    """
    Runs training for a single seed on the fixed split.
    Returns the best validation predictions (probabilities).
    """
    print(f"\n{'='*20} Running Seed {seed} {'='*20}")

    # Set seed for this run
    seed_everything(seed)

    # Datasets
    train_dataset = AppleDataset(df_train, transform=get_transforms("train"))
    val_dataset = AppleDataset(df_val, transform=get_transforms("valid"))

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Model, Optimizer, Scheduler, Criterion
    model = get_model(Config.model_name, Config.pretrained, Config.num_classes)
    model.to(Config.device)

    optimizer = optim.Adam(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )

    # Synced with total epochs
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.epochs, T_mult=Config.T_mult, eta_min=Config.min_lr
    )

    criterion = WeightedSoftTargetCrossEntropy(weight=class_weights)

    # Verify Initial Loss
    print("Verifying initial loss...")
    verify_initial_loss(model, train_loader, criterion, Config.device)

    # Training Loop
    best_auc = 0.0
    best_model_path = os.path.join(
        Config.checkpoint_dir, f"{Config.model_name}_seed_{seed}.pth"
    )

    # To store best predictions
    best_val_preds = None

    for epoch in range(Config.epochs):
        start_time = time.time()

        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, Config.device
        )
        # Validate
        # We need predictions to return, so we modify validate to return them or run inference again
        # For efficiency, we can just use the validate function as is for monitoring,
        # and re-run inference on best model at the end, OR modify validate.
        # Let's modify validate to return preds.
        model.eval()
        running_loss = 0.0
        dataset_size = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(Config.device)
                targets = targets.to(Config.device)
                batch_size = images.size(0)
                outputs = model(images)
                loss = criterion(outputs, targets)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size
                probs = torch.softmax(outputs, dim=1)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        val_loss = running_loss / dataset_size
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        try:
            val_auc = roc_auc_score(all_targets, all_preds, average="macro")
        except ValueError:
            val_auc = 0.5

        # Step scheduler
        scheduler.step()

        elapsed = time.time() - start_time

        # Print metrics
        print(f"Epoch {epoch+1}/{Config.epochs} - Time: {elapsed:.2f}s")
        print(f"  Train Loss: {train_loss}")
        print(f"  Val Loss: {val_loss}")
        print(f"  Val AUC: {val_auc}")

        # Checkpointing
        if val_auc > best_auc:
            print(
                f"  [Improvement] AUC increased from {best_auc} to {val_auc}. Saving model..."
            )
            best_auc = val_auc
            torch.save(model.state_dict(), best_model_path)
            best_val_preds = all_preds

    print(f"Seed {seed} finished. Best AUC: {best_auc}")
    return best_val_preds


# Removed generate_submission and train_and_predict from trainer.py
# as they are better handled in inference.py and runfile.py for this structure.

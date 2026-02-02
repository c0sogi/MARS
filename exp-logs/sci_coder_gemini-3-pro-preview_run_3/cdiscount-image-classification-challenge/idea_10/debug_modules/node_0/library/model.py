import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

from library.config import Config
from library.datasets import FeatureDataset
from library.utils import HierarchyMapper
from library.feature_extractor import extract_and_save_features


class HierarchicalMLP(nn.Module):
    """
    Multi-Task MLP that predicts product categories at three hierarchical levels.
    """

    def __init__(self):
        super(HierarchicalMLP, self).__init__()

        # Shared Trunk
        self.trunk = nn.Sequential(
            nn.Linear(Config.INPUT_DIM, Config.HIDDEN_LAYERS[0]),
            nn.BatchNorm1d(Config.HIDDEN_LAYERS[0]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
            nn.Linear(Config.HIDDEN_LAYERS[0], Config.HIDDEN_LAYERS[1]),
            nn.BatchNorm1d(Config.HIDDEN_LAYERS[1]),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT_RATE),
        )

        # Hierarchical Heads
        self.head_l1 = nn.Linear(Config.HIDDEN_LAYERS[1], Config.NUM_CLASSES_L1)
        self.head_l2 = nn.Linear(Config.HIDDEN_LAYERS[1], Config.NUM_CLASSES_L2)
        self.head_l3 = nn.Linear(Config.HIDDEN_LAYERS[1], Config.NUM_CLASSES_L3)

    def forward(self, x):
        # x shape: (Batch_Size, 3328)
        feat = self.trunk(x)

        logits_l1 = self.head_l1(feat)
        logits_l2 = self.head_l2(feat)
        logits_l3 = self.head_l3(feat)

        return logits_l1, logits_l2, logits_l3


def mixup_data(x, y1, y2, y3, alpha=1.0, device="cuda"):
    """
    Applies MixUp to feature vectors and labels.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]

    # Pairs of labels
    y1_a, y1_b = y1, y1[index]
    y2_a, y2_b = y2, y2[index]
    y3_a, y3_b = y3, y3[index]

    return mixed_x, y1_a, y1_b, y2_a, y2_b, y3_a, y3_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Calculates the MixUp loss.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_one_epoch(model, loader, optimizer, criterion, device, alpha):
    model.train()
    running_loss = 0.0
    correct_l3 = 0
    total = 0

    # Use tqdm for progress tracking if verbose, otherwise silent
    pbar = tqdm(loader, desc="Training", leave=False, disable=True)

    for features, l1, l2, l3 in loader:
        features = features.to(device)
        l1 = l1.to(device)
        l2 = l2.to(device)
        l3 = l3.to(device)

        # Apply MixUp
        features, l1_a, l1_b, l2_a, l2_b, l3_a, l3_b, lam = mixup_data(
            features, l1, l2, l3, alpha=alpha, device=device
        )

        optimizer.zero_grad()

        # Forward pass
        p1, p2, p3 = model(features)

        # Multi-Task Loss with MixUp
        loss_l1 = mixup_criterion(criterion, p1, l1_a, l1_b, lam)
        loss_l2 = mixup_criterion(criterion, p2, l2_a, l2_b, lam)
        loss_l3 = mixup_criterion(criterion, p3, l3_a, l3_b, lam)

        total_loss = loss_l1 + loss_l2 + loss_l3

        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item() * features.size(0)

        # Calculate accuracy on the primary target (approximate for MixUp)
        _, predicted = torch.max(p3.data, 1)
        # Compare against the dominant label in the mix
        target = l3_a if lam > 0.5 else l3_b
        total += l3.size(0)
        correct_l3 += (predicted == target).sum().item()

        pbar.update(1)

    epoch_loss = running_loss / total
    epoch_acc = correct_l3 / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total = 0

    with torch.no_grad():
        for features, l1, l2, l3 in loader:
            features = features.to(device)
            l1 = l1.to(device)
            l2 = l2.to(device)
            l3 = l3.to(device)

            p1, p2, p3 = model(features)

            # Standard Loss (No MixUp)
            loss_l1 = criterion(p1, l1)
            loss_l2 = criterion(p2, l2)
            loss_l3 = criterion(p3, l3)
            total_loss = loss_l1 + loss_l2 + loss_l3

            running_loss += total_loss.item() * features.size(0)

            # Accuracy (L3)
            _, predicted = torch.max(p3.data, 1)
            total += l3.size(0)
            correct_l3 += (predicted == l3).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct_l3 / total
    return epoch_loss, epoch_acc


def train_model(train_loader, val_loader):
    """
    Main training loop with Early Stopping and Checkpointing.
    """
    device = torch.device(Config.DEVICE)
    print(f"Training on device: {device}")

    model = HierarchicalMLP().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    # Loss Function (Label Smoothing helps with noisy/fine-grained labels)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(Config.NUM_EPOCHS):
        start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, Config.MIXUP_ALPHA
        )

        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Step scheduler based on validation accuracy
        scheduler.step(val_acc)

        duration = time.time() - start_time
        print(
            f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
            f"Time: {duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc (L3): {val_acc:.6f}"
        )

        # Checkpointing
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            print(f"  -> New best model saved! Acc: {best_acc:.6f}")
        else:
            patience_counter += 1
            print(
                f"  -> No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
            )

        # Early Stopping
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print("Early stopping triggered.")
            break

    # Load best model for return
    model.load_state_dict(torch.load(Config.MODEL_CHECKPOINT))
    return model


def generate_submission(model, test_loader, hierarchy_mapper):
    """
    Generates predictions for the test set and saves to CSV.
    """
    device = torch.device(Config.DEVICE)
    model.eval()

    predictions = []

    print("Generating predictions on test set...")
    with torch.no_grad():
        for features, ids in tqdm(test_loader, desc="Inference"):
            features = features.to(device)

            # Forward pass
            _, _, p3 = model(features)

            # Get predicted class indices
            _, predicted_indices = torch.max(p3, 1)

            # Convert to CPU numpy
            predicted_indices = predicted_indices.cpu().numpy()
            ids = ids.numpy()

            for _id, pred_idx in zip(ids, predicted_indices):
                # Map L3 index back to category_id
                category_id = hierarchy_mapper.get_category_id(pred_idx)
                predictions.append({"_id": _id, "category_id": category_id})

    # Create DataFrame
    df = pd.DataFrame(predictions)

    # Save submission
    print(f"Saving submission to {Config.SUBMISSION_PATH}...")
    df.to_csv(Config.SUBMISSION_PATH, index=False)
    print("Submission saved successfully.")


def run(load_cached_data=True):
    """
    Orchestrates the full pipeline:
    1. Feature Extraction (if needed)
    2. Data Loading
    3. Training
    4. Inference
    """
    # 1. Feature Extraction
    # This checks cache and extracts features from BSON if missing
    extract_and_save_features(
        load_cached_data=load_cached_data,
        subset_size=Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None,
    )

    # 2. Prepare Datasets
    print("Initializing datasets...")

    # Hierarchy Mapper needed for decoding predictions
    mapper = HierarchyMapper()
    mapper.process()

    train_dataset = FeatureDataset(
        features_path=Config.TRAIN_FEATURES,
        labels_path=Config.TRAIN_LABELS,
        hierarchy_mapper=mapper,
        mode="train",
        subset_size=Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None,
    )

    val_dataset = FeatureDataset(
        features_path=Config.VAL_FEATURES,
        labels_path=Config.VAL_LABELS,
        hierarchy_mapper=mapper,
        mode="val",
        subset_size=Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None,
    )

    test_dataset = FeatureDataset(
        features_path=Config.TEST_FEATURES,
        ids_path=Config.TEST_IDS,
        mode="test",
        subset_size=Config.DEBUG_SUBSET_SIZE if Config.DEBUG else None,
    )

    # DataLoaders
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Train
    model = train_model(train_loader, val_loader)

    # 4. Inference
    generate_submission(model, test_loader, mapper)

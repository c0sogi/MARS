import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.config import (
    EMBEDDING_DIM,
    BATCH_SIZE_TRAIN,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    PATIENCE,
    DEVICE,
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    MODEL_PATH,
    SUBMISSION_PATH,
    SEED,
)
from library.data_utils import seed_everything, HierarchyEncoder
from library.dataset import DecoupledFeatureDataset


class HierarchicalMLP(nn.Module):
    """
    A Multi-Task MLP that predicts Level 1, Level 2, and Level 3 categories
    from a shared feature embedding.
    """

    def __init__(self, input_dim, num_l1, num_l2, num_l3, hidden_dim=1024, dropout=0.3):
        super(HierarchicalMLP, self).__init__()

        # Shared Trunk
        # Input (1280) -> Hidden (1024) -> Hidden (512)
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        feature_dim = hidden_dim // 2

        # Independent Classification Heads
        self.head_l1 = nn.Linear(feature_dim, num_l1)
        self.head_l2 = nn.Linear(feature_dim, num_l2)
        self.head_l3 = nn.Linear(feature_dim, num_l3)  # Primary Target

    def forward(self, x):
        # Shared features
        features = self.trunk(x)

        # Task-specific logits
        l1_logits = self.head_l1(features)
        l2_logits = self.head_l2(features)
        l3_logits = self.head_l3(features)

        return l1_logits, l2_logits, l3_logits


def train_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    for features, (l1_target, l2_target, l3_target) in loader:
        features = features.to(device)
        l1_target = l1_target.to(device)
        l2_target = l2_target.to(device)
        l3_target = l3_target.to(device)

        optimizer.zero_grad()

        l1_logits, l2_logits, l3_logits = model(features)

        # Multi-task Loss: Sum of Cross Entropies
        loss_l1 = F.cross_entropy(l1_logits, l1_target)
        loss_l2 = F.cross_entropy(l2_logits, l2_target)
        loss_l3 = F.cross_entropy(l3_logits, l3_target)

        total_loss = loss_l1 + loss_l2 + loss_l3

        total_loss.backward()
        optimizer.step()

        # Track metrics
        batch_size = features.size(0)
        running_loss += total_loss.item() * batch_size

        # We focus on L3 accuracy for monitoring
        preds_l3 = torch.argmax(l3_logits, dim=1)
        correct_l3 += (preds_l3 == l3_target).sum().item()
        total_samples += batch_size

    avg_loss = running_loss / total_samples
    acc_l3 = correct_l3 / total_samples

    return avg_loss, acc_l3


def validate(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct_l3 = 0
    total_samples = 0

    with torch.no_grad():
        for features, (l1_target, l2_target, l3_target) in loader:
            features = features.to(device)
            l1_target = l1_target.to(device)
            l2_target = l2_target.to(device)
            l3_target = l3_target.to(device)

            l1_logits, l2_logits, l3_logits = model(features)

            loss_l1 = F.cross_entropy(l1_logits, l1_target)
            loss_l2 = F.cross_entropy(l2_logits, l2_target)
            loss_l3 = F.cross_entropy(l3_logits, l3_target)

            total_loss = loss_l1 + loss_l2 + loss_l3

            batch_size = features.size(0)
            running_loss += total_loss.item() * batch_size

            preds_l3 = torch.argmax(l3_logits, dim=1)
            correct_l3 += (preds_l3 == l3_target).sum().item()
            total_samples += batch_size

    avg_loss = running_loss / total_samples
    acc_l3 = correct_l3 / total_samples

    return avg_loss, acc_l3


def train_model(
    train_features_path=TRAIN_FEATURES_PATH,
    train_labels_path=TRAIN_LABELS_PATH,
    val_features_path=VAL_FEATURES_PATH,
    val_labels_path=VAL_LABELS_PATH,
    model_save_path=MODEL_PATH,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE_TRAIN,
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    patience=PATIENCE,
    device=DEVICE,
):
    """
    Main training routine.
    """
    seed_everything(SEED)

    print("Initializing Hierarchy Encoder...")
    encoder = HierarchyEncoder(load_cached_data=True)

    print("Loading Datasets...")
    train_dataset = DecoupledFeatureDataset(
        train_features_path, train_labels_path, hierarchy_encoder=encoder, is_test=False
    )
    val_dataset = DecoupledFeatureDataset(
        val_features_path, val_labels_path, hierarchy_encoder=encoder, is_test=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Initialize Model
    model = HierarchicalMLP(
        input_dim=EMBEDDING_DIM,
        num_l1=encoder.num_l1,
        num_l2=encoder.num_l2,
        num_l3=encoder.num_l3,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=False
    )

    best_val_acc = 0.0
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, device)

        # Print full precision metrics as requested
        print(
            f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Train L3 Acc: {train_acc} - Val Loss: {val_loss} - Val L3 Acc: {val_acc}"
        )

        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved to {model_save_path}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training finished. Best Val L3 Acc: {best_val_acc}")


def predict_submission(
    test_features_path=TEST_FEATURES_PATH,
    test_ids_path=TEST_IDS_PATH,
    model_path=MODEL_PATH,
    submission_path=SUBMISSION_PATH,
    batch_size=BATCH_SIZE_TRAIN,
    device=DEVICE,
):
    """
    Loads the best model, predicts on test set, and saves submission file.
    """
    seed_everything(SEED)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path}. Train model first."
        )

    print("Initializing Hierarchy Encoder...")
    encoder = HierarchyEncoder(load_cached_data=True)

    print("Loading Test Dataset...")
    test_dataset = DecoupledFeatureDataset(
        test_features_path, test_ids_path, is_test=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    print("Loading Model...")
    model = HierarchicalMLP(
        input_dim=EMBEDDING_DIM,
        num_l1=encoder.num_l1,
        num_l2=encoder.num_l2,
        num_l3=encoder.num_l3,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    all_ids = []
    all_preds = []

    print("Running Inference...")
    with torch.no_grad():
        for features, product_ids in test_loader:
            features = features.to(device)

            _, _, l3_logits = model(features)

            # Get predicted L3 index
            preds_idx = torch.argmax(l3_logits, dim=1).cpu().numpy()

            # Map back to category_id
            # We can vectorize the lookup if needed, but list comp is fine for batch
            batch_cat_ids = [encoder.get_category_id(idx) for idx in preds_idx]

            all_ids.extend(product_ids.numpy())
            all_preds.extend(batch_cat_ids)

    # Create Submission DataFrame
    df_sub = pd.DataFrame({"_id": all_ids, "category_id": all_preds})

    # Ensure _id is int
    df_sub["_id"] = df_sub["_id"].astype(int)
    df_sub["category_id"] = df_sub["category_id"].astype(int)

    os.makedirs(os.path.dirname(submission_path), exist_ok=True)
    df_sub.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path} with {len(df_sub)} rows.")

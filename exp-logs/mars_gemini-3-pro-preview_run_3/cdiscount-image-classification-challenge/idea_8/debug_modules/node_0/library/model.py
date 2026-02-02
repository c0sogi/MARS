import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
import time
from tqdm import tqdm

from library.config import Config
from library.dataset import FeatureDataset, MixupCollate
from library.data_utils import HierarchyMapper


class HierarchicalMultiTaskNetwork(nn.Module):
    """
    A Multi-Task Neural Network based on a ResNet-50 backbone features.
    Consists of a shared trunk and three specific heads for hierarchical classification.
    """

    def __init__(self):
        super(HierarchicalMultiTaskNetwork, self).__init__()

        # Shared Trunk
        # Input: 2048 (ResNet50 feature dim) -> Output: 1024
        self.trunk = nn.Sequential(
            nn.Linear(Config.FEATURE_DIM, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(Config.DROPOUT),
        )

        # Level 1 Head (Coarse Category)
        self.head_l1 = nn.Linear(1024, Config.NUM_CLASSES_L1)

        # Level 2 Head (Sub-Category)
        self.head_l2 = nn.Linear(1024, Config.NUM_CLASSES_L2)

        # Level 3 Head (Fine-Grained Category - Target)
        self.head_l3 = nn.Linear(1024, Config.NUM_CLASSES_L3)

    def forward(self, x):
        # x shape: (Batch, 2048)
        embedding = self.trunk(x)

        # Parallel Heads
        logits_l1 = self.head_l1(embedding)
        logits_l2 = self.head_l2(embedding)
        logits_l3 = self.head_l3(embedding)

        return logits_l1, logits_l2, logits_l3


class HierarchicalTrainer:
    """
    Handles training, validation, and inference for the Hierarchical Network.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.device = device if device else Config.DEVICE
        self.model.to(self.device)

        # Hyperparameters
        self.lr = Config.LEARNING_RATE
        self.weight_decay = Config.WEIGHT_DECAY
        self.patience = Config.PATIENCE

        # Loss Weights
        self.w_l1 = Config.WEIGHT_L1
        self.w_l2 = Config.WEIGHT_L2
        self.w_l3 = Config.WEIGHT_L3

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, verbose=True
        )

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Unpack batch from MixupCollate
            # mixed_features: (B, 2048)
            # targets_*: (B, Num_Classes) - Soft targets (probabilities)
            features, y1_soft, y2_soft, y3_soft = batch

            features = features.to(self.device)
            y1_soft = y1_soft.to(self.device)
            y2_soft = y2_soft.to(self.device)
            y3_soft = y3_soft.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            pred_l1, pred_l2, pred_l3 = self.model(features)

            # Multi-Task Loss with Soft Targets
            # CrossEntropyLoss supports prob targets (since PyTorch 1.10)
            loss_l1 = F.cross_entropy(pred_l1, y1_soft)
            loss_l2 = F.cross_entropy(pred_l2, y2_soft)
            loss_l3 = F.cross_entropy(pred_l3, y3_soft)

            loss = (self.w_l3 * loss_l3) + (self.w_l2 * loss_l2) + (self.w_l1 * loss_l1)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in dataloader:
                # Validation uses standard collate: (features, labels)
                # labels: (B, 3) -> [l1, l2, l3]
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)

                # We only care about L3 accuracy for the metric
                target_l3 = labels[:, 2]

                _, _, pred_l3 = self.model(features)

                # Hard predictions
                predicted_classes = torch.argmax(pred_l3, dim=1)

                correct += (predicted_classes == target_l3).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0
        return accuracy

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        best_acc = 0.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(train_loader)
            val_acc = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step(val_acc)

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Acc: {val_acc:.10f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"New best model saved with accuracy: {best_acc:.10f}")
            else:
                patience_counter += 1

            # Early Stopping
            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Accuracy: {best_acc:.10f}")

        # Load best weights
        self.model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH))

    def predict_submission(self, test_features_path, test_ids_path, output_path):
        print("Generating submission...")
        self.model.eval()

        # Load Test Data
        # We load into memory for speed as per design
        test_dataset = FeatureDataset(
            test_features_path, labels_path=None, load_in_memory=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
        )

        test_ids = np.load(test_ids_path)

        all_preds = []

        with torch.no_grad():
            for features in test_loader:
                features = features.to(self.device)
                _, _, logits_l3 = self.model(features)
                preds = torch.argmax(logits_l3, dim=1).cpu().numpy()
                all_preds.append(preds)

        final_preds_idx = np.concatenate(all_preds)

        # Map indices back to category_ids
        mapper = HierarchyMapper(load_cached_data=True)

        # Vectorized mapping is faster, but dictionary lookup is safer for arbitrary IDs
        # We'll use a list comprehension
        final_category_ids = [mapper.get_category_id(idx) for idx in final_preds_idx]

        # Create DataFrame
        submission_df = pd.DataFrame(
            {"_id": test_ids, "category_id": final_category_ids}
        )

        # Ensure integer types
        submission_df["_id"] = submission_df["_id"].astype(int)
        submission_df["category_id"] = submission_df["category_id"].astype(int)

        submission_df.to_csv(output_path, index=False)
        print(f"Submission saved to {output_path} with {len(submission_df)} records.")


def run_training_pipeline():
    """
    Orchestrates the entire training and submission process.
    """
    # 1. Setup Data
    print("Setting up datasets...")

    # Train Dataset with MixUp
    train_ds = FeatureDataset(
        Config.TRAIN_FEATURES_PATH,
        Config.TRAIN_LABELS_PATH,
        limit=Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None,
    )

    # MixUp Collate
    mixup_collate = MixupCollate(alpha=Config.MIXUP_ALPHA)

    train_loader = DataLoader(
        train_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=mixup_collate,
        pin_memory=True,
    )

    # Val Dataset (Standard)
    val_ds = FeatureDataset(
        Config.VAL_FEATURES_PATH,
        Config.VAL_LABELS_PATH,
        limit=Config.DEBUG_SAMPLE_SIZE if Config.DEBUG else None,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 2. Initialize Model & Trainer
    model = HierarchicalMultiTaskNetwork()
    trainer = HierarchicalTrainer(model)

    # 3. Train
    trainer.fit(train_loader, val_loader, epochs=Config.EPOCHS)

    # 4. Predict
    trainer.predict_submission(
        Config.TEST_FEATURES_PATH, Config.TEST_IDS_PATH, Config.SUBMISSION_PATH
    )

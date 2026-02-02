import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import (
    TRAIN_FEATURES_PATH,
    TRAIN_LABELS_PATH,
    VAL_FEATURES_PATH,
    VAL_LABELS_PATH,
    TEST_FEATURES_PATH,
    TEST_IDS_PATH,
    MODEL_PATH,
    SUBMISSION_PATH,
    DEVICE,
    SEED,
    BATCH_SIZE_TRAIN,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EMBEDDING_DIM,
    NUM_EPOCHS,
    PATIENCE,
)
from library.config import seed_everything
from library.data_utils import HierarchyEncoder
from library.dataset import DecoupledFeatureDataset
from library.model import HierarchicalMLP


class Trainer:
    """
    Manages the training, validation, and inference lifecycle of the Hierarchical Multi-Task Network.
    """

    def __init__(
        self,
        train_features_path=TRAIN_FEATURES_PATH,
        train_labels_path=TRAIN_LABELS_PATH,
        val_features_path=VAL_FEATURES_PATH,
        val_labels_path=VAL_LABELS_PATH,
        model_save_path=MODEL_PATH,
        batch_size=BATCH_SIZE_TRAIN,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        device=DEVICE,
    ):

        self.device = device
        self.model_save_path = model_save_path
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay

        # Ensure reproducibility
        seed_everything(SEED)

        # 1. Prepare Metadata & Encoder
        print("Initializing Hierarchy Encoder...")
        self.encoder = HierarchyEncoder(load_cached_data=True)

        # 2. Prepare Datasets & Loaders
        print("Loading Datasets...")
        self.train_dataset = DecoupledFeatureDataset(
            train_features_path,
            train_labels_path,
            hierarchy_encoder=self.encoder,
            is_test=False,
        )
        self.val_dataset = DecoupledFeatureDataset(
            val_features_path,
            val_labels_path,
            hierarchy_encoder=self.encoder,
            is_test=False,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # 3. Initialize Model
        # We use the encoder to determine the number of classes for each head
        self.model = HierarchicalMLP(
            input_dim=EMBEDDING_DIM,
            num_l1=self.encoder.num_l1,
            num_l2=self.encoder.num_l2,
            num_l3=self.encoder.num_l3,
        ).to(self.device)

        # 4. Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

        # 5. Loss Function
        # Multi-task learning: Sum of CrossEntropy losses.
        # We do not use class weights to prioritize global accuracy.
        self.criterion = nn.CrossEntropyLoss()

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        correct_l3 = 0
        total_samples = 0

        for features, (l1_target, l2_target, l3_target) in self.train_loader:
            features = features.to(self.device)
            l1_target = l1_target.to(self.device)
            l2_target = l2_target.to(self.device)
            l3_target = l3_target.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            l1_logits, l2_logits, l3_logits = self.model(features)

            # Compute losses for all levels
            loss_l1 = self.criterion(l1_logits, l1_target)
            loss_l2 = self.criterion(l2_logits, l2_target)
            loss_l3 = self.criterion(l3_logits, l3_target)

            # Sum losses (unweighted)
            total_loss = loss_l1 + loss_l2 + loss_l3

            # Backward pass
            total_loss.backward()
            self.optimizer.step()

            # Metrics
            batch_size = features.size(0)
            running_loss += total_loss.item() * batch_size

            # Track L3 accuracy (primary target)
            preds_l3 = torch.argmax(l3_logits, dim=1)
            correct_l3 += (preds_l3 == l3_target).sum().item()
            total_samples += batch_size

        avg_loss = running_loss / total_samples
        acc_l3 = correct_l3 / total_samples
        return avg_loss, acc_l3

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct_l3 = 0
        total_samples = 0

        with torch.no_grad():
            for features, (l1_target, l2_target, l3_target) in self.val_loader:
                features = features.to(self.device)
                l1_target = l1_target.to(self.device)
                l2_target = l2_target.to(self.device)
                l3_target = l3_target.to(self.device)

                l1_logits, l2_logits, l3_logits = self.model(features)

                loss_l1 = self.criterion(l1_logits, l1_target)
                loss_l2 = self.criterion(l2_logits, l2_target)
                loss_l3 = self.criterion(l3_logits, l3_target)

                total_loss = loss_l1 + loss_l2 + loss_l3

                batch_size = features.size(0)
                running_loss += total_loss.item() * batch_size

                preds_l3 = torch.argmax(l3_logits, dim=1)
                correct_l3 += (preds_l3 == l3_target).sum().item()
                total_samples += batch_size

        avg_loss = running_loss / total_samples
        acc_l3 = correct_l3 / total_samples
        return avg_loss, acc_l3

    def fit(self, epochs=NUM_EPOCHS, patience=PATIENCE):
        """
        Main training loop with Early Stopping.
        """
        best_val_acc = -1.0
        patience_counter = 0

        print(f"Starting Training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            # Print full precision metrics
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Train L3 Acc: {train_acc} - Val Loss: {val_loss} - Val L3 Acc: {val_acc}"
            )

            # Update Scheduler
            self.scheduler.step(val_acc)

            # Checkpoint & Early Stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                # Save Model
                os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved to {self.model_save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Val L3 Acc: {best_val_acc}")

    def predict(
        self,
        test_features_path=TEST_FEATURES_PATH,
        test_ids_path=TEST_IDS_PATH,
        submission_path=SUBMISSION_PATH,
    ):
        """
        Generates predictions for the test set using the best saved model.
        """
        print("Starting Inference...")

        # Load Test Data
        test_dataset = DecoupledFeatureDataset(
            test_features_path, test_ids_path, is_test=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        # Load Best Model
        if not os.path.exists(self.model_save_path):
            raise FileNotFoundError(
                f"Model file not found at {self.model_save_path}. Train model first."
            )

        print(f"Loading model from {self.model_save_path}...")
        self.model.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds = []

        with torch.no_grad():
            for features, product_ids in test_loader:
                features = features.to(self.device)

                # Forward pass
                _, _, l3_logits = self.model(features)

                # Get predicted L3 index
                preds_idx = torch.argmax(l3_logits, dim=1).cpu().numpy()

                # Map back to category_id using the encoder
                batch_cat_ids = [self.encoder.get_category_id(idx) for idx in preds_idx]

                all_ids.extend(product_ids.numpy())
                all_preds.extend(batch_cat_ids)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"_id": all_ids, "category_id": all_preds})

        # Ensure types
        df_sub["_id"] = df_sub["_id"].astype(int)
        df_sub["category_id"] = df_sub["category_id"].astype(int)

        # Save
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path} with {len(df_sub)} rows.")

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import (
    MODEL_SAVE_PATH,
    DEVICE,
    SEED,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    SUBMISSION_PATH,
    INPUT_DIM,
    HIDDEN_DIM,
    DROPOUT_RATE,
    NUM_CLASSES_L1,
    NUM_CLASSES_L2,
    NUM_CLASSES_L3,
    BATCH_SIZE,
    USE_MIXUP,
    MIXUP_ALPHA,
    NUM_WORKERS,
)
from library.data_utils import HierarchyManager
from library.feature_dataset import get_dataloaders
from library.cascade_model import ConditionalCascadeMLP, HierarchicalLoss


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the
    Conditional Cascade MLP model.
    """

    def __init__(self, hierarchy_manager, model_save_path=MODEL_SAVE_PATH):
        self.hierarchy_manager = hierarchy_manager
        self.model_save_path = model_save_path
        self.device = torch.device(DEVICE)

        # Initialize Model
        self.model = ConditionalCascadeMLP(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout_rate=DROPOUT_RATE,
            num_classes_l1=NUM_CLASSES_L1,
            num_classes_l2=NUM_CLASSES_L2,
            num_classes_l3=NUM_CLASSES_L3,
        ).to(self.device)

        # Loss and Optimizer
        self.criterion = HierarchicalLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        total_samples = 0

        for batch_idx, (features, targets_a, targets_b, lam) in enumerate(train_loader):
            features = features.to(self.device)

            # Unpack targets and move to device
            l1_a, l2_a, l3_a = [t.to(self.device) for t in targets_a]
            l1_b, l2_b, l3_b = [t.to(self.device) for t in targets_b]

            targets_a_device = (l1_a, l2_a, l3_a)
            targets_b_device = (l1_b, l2_b, l3_b)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)

            # Compute loss
            loss = self.criterion(outputs, targets_a_device, targets_b_device, lam)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

        return running_loss / total_samples

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns accuracy for all three levels.
        """
        self.model.eval()
        correct_l1 = 0
        correct_l2 = 0
        correct_l3 = 0
        total = 0

        with torch.no_grad():
            for features, targets_a, _, _ in val_loader:
                features = features.to(self.device)

                # Validation loader does not use MixUp, so targets_a == targets_b
                l1_target, l2_target, l3_target = [t.to(self.device) for t in targets_a]

                logits_l1, logits_l2, logits_l3 = self.model(features)

                # Predictions
                pred_l1 = torch.argmax(logits_l1, dim=1)
                pred_l2 = torch.argmax(logits_l2, dim=1)
                pred_l3 = torch.argmax(logits_l3, dim=1)

                correct_l1 += (pred_l1 == l1_target).sum().item()
                correct_l2 += (pred_l2 == l2_target).sum().item()
                correct_l3 += (pred_l3 == l3_target).sum().item()
                total += features.size(0)

        acc_l1 = correct_l1 / total
        acc_l2 = correct_l2 / total
        acc_l3 = correct_l3 / total

        return acc_l1, acc_l2, acc_l3

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=NUM_EPOCHS,
        patience=EARLY_STOPPING_PATIENCE,
    ):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training on device: {self.device}")
        best_val_acc_l3 = 0.0
        patience_counter = 0

        for epoch in range(epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_acc_l1, val_acc_l2, val_acc_l3 = self.validate(val_loader)

            # Step scheduler based on L3 accuracy (primary metric)
            self.scheduler.step(val_acc_l3)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch+1}/{epochs} - "
                f"Time: {elapsed:.2f}s - "
                f"Train Loss: {train_loss} - "
                f"Val Acc L1: {val_acc_l1} - "
                f"Val Acc L2: {val_acc_l2} - "
                f"Val Acc L3: {val_acc_l3}"
            )

            # Early Stopping Check
            if val_acc_l3 > best_val_acc_l3:
                best_val_acc_l3 = val_acc_l3
                patience_counter = 0
                torch.save(self.model.state_dict(), self.model_save_path)
                print(f"New best model saved with L3 Accuracy: {best_val_acc_l3}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation L3 Accuracy: {best_val_acc_l3}")

    def predict(self, test_loader, output_csv_path=SUBMISSION_PATH):
        """
        Generates predictions for the test set and saves to CSV.
        """
        print(f"Loading best model from {self.model_save_path}...")
        self.model.load_state_dict(
            torch.load(self.model_save_path, map_location=self.device)
        )
        self.model.eval()

        all_ids = []
        all_preds_l3 = []

        print("Generating predictions...")
        with torch.no_grad():
            for features, ids in test_loader:
                features = features.to(self.device)

                _, _, logits_l3 = self.model(features)
                preds_l3 = torch.argmax(logits_l3, dim=1)

                all_ids.extend(ids.numpy())
                all_preds_l3.extend(preds_l3.cpu().numpy())

        # Map L3 indices back to original category IDs
        print("Mapping predictions to category IDs...")
        final_category_ids = [
            self.hierarchy_manager.get_category_id_from_l3(idx) for idx in all_preds_l3
        ]

        # Create submission DataFrame
        df_sub = pd.DataFrame({"_id": all_ids, "category_id": final_category_ids})

        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)

        df_sub.to_csv(output_csv_path, index=False)
        print(f"Submission saved to {output_csv_path}")


def run_training_pipeline(
    train_features_path,
    train_labels_path,
    val_features_path,
    val_labels_path,
    test_features_path,
    test_ids_path,
    epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
):
    """
    Orchestrates the data loading, training, and prediction process.
    """
    set_seed(SEED)

    # 1. Initialize Hierarchy Manager
    print("Initializing Hierarchy Manager...")
    hierarchy_manager = HierarchyManager(load_cached_data=True)

    # 2. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_features_path=train_features_path,
        train_labels_path=train_labels_path,
        val_features_path=val_features_path,
        val_labels_path=val_labels_path,
        test_features_path=test_features_path,
        test_ids_path=test_ids_path,
        hierarchy_manager=hierarchy_manager,
        batch_size=batch_size,
        mixup_alpha=MIXUP_ALPHA if USE_MIXUP else 0.0,
        num_workers=NUM_WORKERS,
    )

    # 3. Initialize Trainer
    trainer = Trainer(hierarchy_manager)

    # 4. Train
    trainer.fit(train_loader, val_loader, epochs=epochs)

    # 5. Predict
    trainer.predict(test_loader)

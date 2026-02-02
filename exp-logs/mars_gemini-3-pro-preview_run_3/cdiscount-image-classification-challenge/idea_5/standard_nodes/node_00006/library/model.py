import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import os
from library import config, utils, dataset


class HierarchicalMLP(nn.Module):
    """
    A Multi-Task MLP that predicts Level 1, Level 2, and Level 3 categories
    from a shared feature representation.
    """

    def __init__(self, num_l1, num_l2, num_l3):
        super(HierarchicalMLP, self).__init__()

        # Shared feature extractor
        self.shared_net = nn.Sequential(
            nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM),
            nn.BatchNorm1d(config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT_RATE),
        )

        # Task-specific heads
        self.head_l1 = nn.Linear(config.HIDDEN_DIM, num_l1)
        self.head_l2 = nn.Linear(config.HIDDEN_DIM, num_l2)
        self.head_l3 = nn.Linear(config.HIDDEN_DIM, num_l3)

    def forward(self, x):
        features = self.shared_net(x)
        out_l1 = self.head_l1(features)
        out_l2 = self.head_l2(features)
        out_l3 = self.head_l3(features)
        return out_l1, out_l2, out_l3


class HierarchicalTrainer:
    """
    Manages training and inference for the HierarchicalMLP.
    """

    def __init__(self):
        self.device = config.DEVICE

        # Ensure encoder is ready to provide class counts
        self.encoder = utils.HierarchyEncoder()
        self.encoder.prepare()

        self.num_l1 = self.encoder.num_l1
        self.num_l2 = self.encoder.num_l2
        self.num_l3 = self.encoder.num_l3

        print(
            f"Initializing model with L1={self.num_l1}, L2={self.num_l2}, L3={self.num_l3} classes."
        )
        self.model = HierarchicalMLP(self.num_l1, self.num_l2, self.num_l3).to(
            self.device
        )
        self.best_model_path = os.path.join(config.WORKING_DIR, "best_model.pth")

    def train(self):
        """
        Trains the model using pre-computed embeddings.
        """
        print("Loading training and validation datasets...")
        # Check if feature files exist
        if not os.path.exists(config.TRAIN_FEATURES_PATH):
            raise FileNotFoundError(
                f"Train features not found at {config.TRAIN_FEATURES_PATH}. Run feature extraction first."
            )

        train_ds = dataset.EmbeddingDataset(
            config.TRAIN_FEATURES_PATH,
            config.TRAIN_LABELS_L1_PATH,
            config.TRAIN_LABELS_L2_PATH,
            config.TRAIN_LABELS_L3_PATH,
            mode="train",
        )
        val_ds = dataset.EmbeddingDataset(
            config.VAL_FEATURES_PATH,
            config.VAL_LABELS_L1_PATH,
            config.VAL_LABELS_L2_PATH,
            config.VAL_LABELS_L3_PATH,
            mode="val",
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        criterion = nn.CrossEntropyLoss()

        best_val_acc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {config.NUM_EPOCHS} epochs...")

        for epoch in range(config.NUM_EPOCHS):
            self.model.train()
            total_loss = 0.0

            for features, l1, l2, l3 in train_loader:
                features = features.to(self.device)
                l1, l2, l3 = l1.to(self.device), l2.to(self.device), l3.to(self.device)

                optimizer.zero_grad()

                # Forward pass
                out_l1, out_l2, out_l3 = self.model(features)

                # Multi-task loss
                loss_l1 = criterion(out_l1, l1)
                loss_l2 = criterion(out_l2, l2)
                loss_l3 = criterion(out_l3, l3)

                loss = (
                    (config.WEIGHT_L3 * loss_l3)
                    + (config.WEIGHT_L2 * loss_l2)
                    + (config.WEIGHT_L1 * loss_l1)
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_train_loss = total_loss / len(train_loader)

            # Validation
            val_acc = self.evaluate(val_loader)

            print(
                f"Epoch {epoch+1}/{config.NUM_EPOCHS} - Train Loss: {avg_train_loss:.6f} - Val Acc L3: {val_acc:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with accuracy: {val_acc:.6f}")
            else:
                patience_counter += 1
                if patience_counter >= config.PATIENCE:
                    print("Early stopping triggered.")
                    break

    def evaluate(self, loader):
        """
        Evaluates the model on the primary task (Level 3 accuracy).
        """
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for features, _, _, l3 in loader:
                features = features.to(self.device)
                l3 = l3.to(self.device)

                _, _, out_l3 = self.model(features)
                _, predicted = torch.max(out_l3, 1)

                total += l3.size(0)
                correct += (predicted == l3).sum().item()

        return correct / total if total > 0 else 0.0

    def predict_submission(self):
        """
        Generates predictions for the test set and saves the submission CSV.
        """
        print("Generating submission...")

        if not os.path.exists(self.best_model_path):
            print("No model found. Please train first.")
            return

        # Load best weights
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )
        self.model.eval()

        # Check for test features
        if not os.path.exists(config.TEST_FEATURES_PATH):
            raise FileNotFoundError(
                "Test features not found. Run feature extraction first."
            )

        test_ds = dataset.EmbeddingDataset(
            config.TEST_FEATURES_PATH, ids_path=config.TEST_IDS_PATH, mode="test"
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
        )

        all_ids = []
        all_preds = []

        print("Running inference on test set...")
        with torch.no_grad():
            for features, ids in test_loader:
                features = features.to(self.device)

                # Only interested in L3 predictions for submission
                _, _, out_l3 = self.model(features)
                _, predicted_indices = torch.max(out_l3, 1)

                all_ids.extend(ids.numpy())
                all_preds.extend(predicted_indices.cpu().numpy())

        # Decode predictions (Index -> Category ID)
        print("Decoding predictions...")
        category_ids = self.encoder.inverse_transform(all_preds)

        # Create Submission DataFrame
        df = pd.DataFrame({"_id": all_ids, "category_id": category_ids})

        # Save
        submission_path = os.path.join(config.SUBMISSION_DIR, "submission.csv")
        df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

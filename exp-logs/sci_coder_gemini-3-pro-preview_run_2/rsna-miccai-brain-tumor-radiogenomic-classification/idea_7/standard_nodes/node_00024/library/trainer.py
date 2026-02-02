import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.utils import get_device, calculate_roc_auc, seed_everything
from library.dataset import BraTSDataset
from library.model import ModalityGroupedEfficientNet


class Trainer:
    def __init__(
        self,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        device: str = None,
    ):
        """
        Initializes the Trainer with model, optimizer, and criterion.
        """
        self.device = get_device() if device is None else torch.device(device)

        # Initialize Model
        self.model = ModalityGroupedEfficientNet(num_classes=1, pretrained=True)
        self.model.to(self.device)

        # Optimizer: AdamW with aggressive weight decay as per instructions
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Setup working directory for checkpoints
        self.checkpoint_dir = "./working/idea_7"
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def train_one_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        epoch_loss = running_loss / count if count > 0 else 0.0
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs validation and calculates ROC AUC.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * images.size(0)
                count += images.size(0)

                # Apply sigmoid for probabilities
                probs = torch.sigmoid(outputs)

                all_preds.extend(probs.cpu().numpy().flatten())
                all_labels.extend(labels.cpu().numpy().flatten())

        val_loss = running_loss / count if count > 0 else 0.0
        val_auc = calculate_roc_auc(all_labels, all_preds)

        return val_loss, val_auc

    def fit(
        self,
        epochs: int = 20,
        batch_size: int = 16,
        num_workers: int = 2,
        patience: int = 5,
        load_cached_data: bool = True,
    ):
        """
        Main training loop with Early Stopping.
        """
        seed_everything(42)

        # Initialize Datasets and Loaders
        train_dataset = BraTSDataset(split="train", load_cached_data=load_cached_data)
        val_dataset = BraTSDataset(split="val", load_cached_data=load_cached_data)

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        best_auc = 0.0
        epochs_no_improve = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} - Val AUC: {val_auc}"
            )

            # Save best model based on AUC
            if val_auc > best_auc:
                best_auc = val_auc
                epochs_no_improve = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with AUC: {best_auc}")
            else:
                epochs_no_improve += 1

            # Early Stopping
            if epochs_no_improve >= patience:
                print(
                    f"Early stopping triggered after {patience} epochs with no improvement."
                )
                break

        print(f"Training complete. Best Val AUC: {best_auc}")

    def predict_with_tta(
        self, batch_size: int = 16, num_workers: int = 2, load_cached_data: bool = True
    ):
        """
        Generates predictions for the test set using Test-Time Augmentation (TTA).
        TTA: Average of (Original, Horizontal Flip, Vertical Flip).
        """
        # Load best model
        if not os.path.exists(self.best_model_path):
            print(
                "No best model found. Using current model state (warning: might be untrained)."
            )
        else:
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")

        self.model.eval()

        test_dataset = BraTSDataset(split="test", load_cached_data=load_cached_data)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        results = []

        with torch.no_grad():
            for images, brats_ids in test_loader:
                images = images.to(self.device)

                # 1. Original Prediction
                out_orig = self.model(images)
                prob_orig = torch.sigmoid(out_orig)

                # 2. Horizontal Flip (axis 3: W)
                images_h = torch.flip(images, dims=[3])
                out_h = self.model(images_h)
                prob_h = torch.sigmoid(out_h)

                # 3. Vertical Flip (axis 2: H)
                images_v = torch.flip(images, dims=[2])
                out_v = self.model(images_v)
                prob_v = torch.sigmoid(out_v)

                # Average probabilities
                avg_prob = (prob_orig + prob_h + prob_v) / 3.0

                # Store results
                avg_prob_np = avg_prob.cpu().numpy().flatten()
                ids_np = brats_ids.numpy().flatten()

                for bid, pred in zip(ids_np, avg_prob_np):
                    results.append({"BraTS21ID": bid, "MGMT_value": pred})

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Save submission
        submission_dir = "./submission"
        os.makedirs(submission_dir, exist_ok=True)
        submission_path = os.path.join(submission_dir, "submission.csv")

        # Ensure correct column order
        submission_df = submission_df[["BraTS21ID", "MGMT_value"]]
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

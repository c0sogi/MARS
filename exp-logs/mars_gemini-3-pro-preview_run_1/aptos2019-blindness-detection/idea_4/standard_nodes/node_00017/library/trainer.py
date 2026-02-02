import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.utils import (
    seed_everything,
    quadratic_weighted_kappa,
    decode_ordinal_predictions,
)
from library.dataset import create_dataloaders
from library.model import OrdinalModel


class DRTrainer:
    def __init__(
        self,
        experiment_dir="./working/idea_4",
        model_name="efficientnet_b0",
        img_size=256,
        num_classes=4,
        lr=1e-3,
        weight_decay=1e-5,
        epochs=10,
        batch_size=32,
        num_workers=4,
        device=None,
        patience=5,
        seed=42,
    ):
        self.experiment_dir = experiment_dir
        self.img_size = img_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.patience = patience
        self.seed = seed

        # Ensure experiment directory exists
        os.makedirs(self.experiment_dir, exist_ok=True)

        # Set device
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = OrdinalModel(
            model_name=model_name, pretrained=True, num_classes=num_classes
        )
        self.model.to(self.device)

        # Loss Function
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=epochs
        )

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, targets in train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixed Precision Training
            with torch.cuda.amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(self.device)
                # targets are ordinal vectors (Batch, 4)

                outputs = self.model(images)

                all_preds.append(torch.sigmoid(outputs).cpu())
                all_targets.append(targets.cpu())

        # Concatenate
        preds_tensor = torch.cat(all_preds)
        targets_tensor = torch.cat(all_targets)

        # Decode Predictions: Sum probabilities and round
        y_pred = decode_ordinal_predictions(preds_tensor)

        # Decode Targets: Sum ordinal vector to get integer class
        # e.g., [1, 1, 0, 0] -> 2
        y_true = targets_tensor.sum(dim=1).numpy().astype(int)

        # Calculate QWK
        score = quadratic_weighted_kappa(y_true, y_pred)

        return score

    def fit(self, train_csv, val_csv, sample_size=None):
        seed_everything(self.seed)

        # Create DataLoaders
        # We don't need test_loader here, so we can ignore it or just unpack
        train_loader, val_loader, _ = create_dataloaders(
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=val_csv,  # Placeholder, not used in fit
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            img_size=self.img_size,
            sample_size=sample_size,
            seed=self.seed,
        )

        best_score = -float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.experiment_dir, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(self.epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_score = self.validate(val_loader)

            # Step scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{self.epochs} - Train Loss: {train_loss} - Val QWK: {val_score}"
            )

            # Checkpoint and Early Stopping
            if val_score > best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved to {best_model_path}")
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Val QWK: {best_score}")
        return best_score

    def predict_and_submit(
        self, test_csv, submission_path="./submission/submission.csv"
    ):
        """
        Loads the best model, predicts on the test set, and saves the submission file.
        """
        # Load best model
        best_model_path = os.path.join(self.experiment_dir, "best_model.pth")
        if not os.path.exists(best_model_path):
            print(
                "No best model found. Using current model weights (warning: might be untrained or suboptimal)."
            )
        else:
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print(f"Loaded model from {best_model_path}")

        self.model.eval()

        # Create Test DataLoader
        # We pass dummy paths for train/val as we only need test_loader
        _, _, test_loader = create_dataloaders(
            train_csv=test_csv,  # dummy
            val_csv=test_csv,  # dummy
            test_csv=test_csv,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            img_size=self.img_size,
            sample_size=None,  # Always predict on full test set
            seed=self.seed,
        )

        all_preds = []
        all_ids = []

        with torch.no_grad():
            for images, ids in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)

                all_preds.append(torch.sigmoid(outputs).cpu())
                all_ids.extend(ids)

        # Decode predictions
        preds_tensor = torch.cat(all_preds)
        discrete_preds = decode_ordinal_predictions(preds_tensor)

        # Create Submission DataFrame
        df_sub = pd.DataFrame({"id_code": all_ids, "diagnosis": discrete_preds})

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)

        # Save
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")


def run_training(
    train_csv="./metadata/train.csv",
    val_csv="./metadata/val.csv",
    test_csv="./metadata/test.csv",
    epochs=10,
    batch_size=32,
    sample_size=None,
):
    """
    Helper function to instantiate Trainer and run the pipeline.
    """
    trainer = DRTrainer(epochs=epochs, batch_size=batch_size)

    # Train
    trainer.fit(train_csv, val_csv, sample_size=sample_size)

    # Generate Submission
    trainer.predict_and_submit(test_csv)

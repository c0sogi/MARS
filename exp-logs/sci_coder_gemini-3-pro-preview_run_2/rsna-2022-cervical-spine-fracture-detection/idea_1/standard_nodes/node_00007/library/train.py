import os
import torch
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import RSNADataset
from library.model import FractureMILModel
from library.loss import WeightedLogLoss


class Trainer:
    """
    Trainer class for the Cervical Spine Fracture Detection MIL model.
    Handles training loop, validation, early stopping, and submission generation.
    """

    def __init__(self, config=Config, debug=False):
        """
        Args:
            config: Configuration class containing hyperparameters.
            debug (bool): If True, runs on a small subset of data for debugging.
        """
        self.config = config
        self.debug = debug
        self.device = torch.device(config.DEVICE)

        # Initialize Model
        # We use the pretrained 2D CNN backbone defined in library.model
        self.model = FractureMILModel(pretrained=True).to(self.device)

        # Initialize Loss Function
        # WeightedLogLoss handles the class imbalance and specific competition metric
        self.criterion = WeightedLogLoss().to(self.device)

        # Initialize Optimizer and Scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.EPOCHS
        )

        # Initialize Gradient Scaler for Mixed Precision
        self.scaler = torch.amp.GradScaler("cuda")

        # Prepare DataLoaders
        self.train_loader, self.val_loader = self._get_dataloaders()

    def _get_dataloaders(self):
        """
        Loads metadata and creates DataLoaders for training and validation.
        """
        train_df = pd.read_csv(self.config.TRAIN_METADATA_PATH)
        val_df = pd.read_csv(self.config.VAL_METADATA_PATH)

        # Debug mode: Use a tiny subset of data
        if self.debug:
            train_df = train_df.head(self.config.BATCH_SIZE * 2)
            val_df = val_df.head(self.config.BATCH_SIZE * 2)

        # Initialize Datasets
        # load_cached_paths=True ensures we use the parquet cache for speed
        train_dataset = RSNADataset(train_df, self.config, load_cached_paths=True)
        val_dataset = RSNADataset(val_df, self.config, load_cached_paths=True)

        # Initialize DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0

        for i, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            with torch.amp.autocast("cuda"):
                # Forward pass: (Batch, Slices, 1, H, W) -> (Batch, 8)
                outputs = self.model(images)
                # Compute loss
                loss = self.criterion(outputs, labels)

            # Scaled Backward Pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Use autocast for validation as well to save memory/time
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item()

        avg_loss = running_loss / len(self.val_loader)
        return avg_loss

    def fit(self, epochs=None, patience=3):
        """
        Main training loop with Early Stopping.

        Args:
            epochs (int): Number of epochs to train. If None, uses config.EPOCHS.
            patience (int): Number of epochs to wait for improvement before stopping.
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        if self.debug:
            epochs = min(epochs, 2)
            print("Debug mode: Limiting training to 2 epochs.")

        print(f"Starting training for {epochs} epochs on {self.device}...")

        best_val_loss = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(self.config.CACHE_DIR, "best_model.pth")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(epoch)
            val_loss = self.validate()

            # Update learning rate scheduler
            self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(
                    f"Validation loss improved. Saved best model to {best_model_path}"
                )
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Loss: {best_val_loss}")

    def predict_and_submit(self):
        """
        Generates predictions for the test set and saves submission.csv.
        """
        print("Generating submission...")

        # Load Best Model
        best_model_path = os.path.join(self.config.CACHE_DIR, "best_model.pth")
        if os.path.exists(best_model_path):
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {best_model_path}")
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()

        # Load Test Data
        test_df = pd.read_csv(self.config.TEST_METADATA_PATH)
        test_dataset = RSNADataset(test_df, self.config, load_cached_paths=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=True,
        )

        # Dictionary to store predictions: {StudyInstanceUID: {target_name: probability}}
        study_preds = {}
        target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
        study_ids = test_df["StudyInstanceUID"].tolist()

        # Inference Loop
        with torch.no_grad():
            for i, (images, _) in enumerate(test_loader):
                images = images.to(self.device)

                # Use autocast for inference
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)  # Shape: (Batch, 8)

                outputs = outputs.float().cpu().numpy()

                # Map outputs back to StudyInstanceUIDs
                start_idx = i * self.config.BATCH_SIZE
                end_idx = start_idx + images.size(0)
                batch_uids = study_ids[start_idx:end_idx]

                for uid, preds in zip(batch_uids, outputs):
                    study_preds[uid] = {
                        col: float(p) for col, p in zip(target_cols, preds)
                    }

        # Create Submission File
        # We use sample_submission.csv to ensure we have all required row_ids
        sample_sub = pd.read_csv(self.config.SAMPLE_SUBMISSION_PATH)

        def get_prob(row_id):
            """
            Maps a row_id (e.g., '1.2.3_C1') to the predicted probability.
            """
            for target in target_cols:
                # Check if row_id ends with a known target suffix
                suffix = f"_{target}"
                if row_id.endswith(suffix):
                    # Extract StudyInstanceUID
                    study_uid = row_id[: -len(suffix)]
                    if study_uid in study_preds:
                        return study_preds[study_uid][target]
            # Default fallback
            return 0.5

        sample_sub["fractured"] = sample_sub["row_id"].apply(get_prob)

        # Save to disk
        sample_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

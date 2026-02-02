import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config, seed_everything
from library.dataset import HotelDataset, get_transforms
from library.model import HotelResNet
from library.utils import calculate_map5, get_label_encoder


class Trainer:
    """
    Manages the training, validation, and prediction lifecycle for the Hotel ID model.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (torch.device, optional): Device to run the model on. Defaults to Config.DEVICE.
        """
        self.device = device if device else Config.DEVICE

        # Initialize Model
        self.model = HotelResNet(
            n_classes=Config.NUM_CLASSES, pretrained=Config.PRETRAINED
        )
        self.model.to(self.device)

        # Loss and Optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Checkpoint paths
        os.makedirs(Config.MODEL_CHECKPOINT_DIR, exist_ok=True)
        self.best_model_path = Config.BEST_MODEL_PATH

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training dataloader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = len(dataloader.dataset)

        for batch in dataloader:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, dataloader):
        """
        Runs validation on the provided dataloader.

        Args:
            dataloader (DataLoader): Validation dataloader.

        Returns:
            tuple: (Average Validation Loss, MAP@5 Score)
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = len(dataloader.dataset)

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                targets = batch["target"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                # Get Top-K predictions for MAP@5 calculation
                # outputs shape: (batch_size, num_classes)
                # indices shape: (batch_size, k)
                _, indices = torch.topk(outputs, k=Config.TOP_K, dim=1)

                all_preds.extend(indices.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        val_loss = running_loss / dataset_size
        val_map5 = calculate_map5(all_preds, all_targets)

        return val_loss, val_map5

    def fit(
        self,
        train_metadata_path=Config.TRAIN_METADATA_PATH,
        val_metadata_path=Config.VAL_METADATA_PATH,
        epochs=Config.EPOCHS,
        batch_size=Config.BATCH_SIZE,
        debug=Config.DEBUG,
        num_workers=Config.NUM_WORKERS,
    ):
        """
        Executes the training pipeline with Early Stopping.

        Args:
            train_metadata_path (str): Path to training metadata CSV.
            val_metadata_path (str): Path to validation metadata CSV.
            epochs (int): Maximum number of epochs.
            batch_size (int): Batch size.
            debug (bool): If True, uses a small subset of data.
            num_workers (int): Number of dataloader workers.
        """
        seed_everything(Config.SEED)

        # Load Metadata
        train_df = pd.read_csv(train_metadata_path)
        val_df = pd.read_csv(val_metadata_path)

        # Determine dataset size limit for debugging
        max_size = Config.DEBUG_SAMPLE_SIZE if debug else None

        # Initialize Datasets
        # Note: train_dataset will fit and cache the label encoder
        train_dataset = HotelDataset(
            df=train_df,
            phase="train",
            transform=get_transforms("train"),
            max_size=max_size,
        )

        val_dataset = HotelDataset(
            df=val_df,
            phase="val",
            transform=get_transforms("val"),
            label_encoder=train_dataset.label_encoder,  # Share the fitted encoder
            max_size=max_size,
        )

        # Initialize Dataloaders
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

        # Training Loop
        best_map5 = 0.0
        patience_counter = 0

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_map5 = self.validate(val_loader)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}: Train Loss: {train_loss}, Val Loss: {val_loss}, Val MAP@5: {val_map5}"
            )

            # Early Stopping and Checkpointing
            if val_map5 > best_map5:
                best_map5 = val_map5
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val MAP@5: {best_map5}")

    def predict(
        self,
        test_metadata_path=Config.TEST_METADATA_PATH,
        submission_path=Config.SUBMISSION_PATH,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    ):
        """
        Generates predictions for the test set and saves the submission file.

        Args:
            test_metadata_path (str): Path to test metadata CSV.
            submission_path (str): Path to save the submission CSV.
            batch_size (int): Batch size.
            num_workers (int): Number of dataloader workers.
        """
        # Load Test Metadata
        test_df = pd.read_csv(test_metadata_path)

        # Load Label Encoder (must be cached from training phase)
        label_encoder = get_label_encoder(load_cached_data=True)

        # Initialize Test Dataset
        test_dataset = HotelDataset(
            df=test_df,
            phase="test",
            transform=get_transforms("test"),
            label_encoder=label_encoder,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Load Best Model Weights
        if os.path.exists(self.best_model_path):
            state_dict = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()
        results = []

        # Inference Loop
        with torch.no_grad():
            for batch in test_loader:
                images = batch["image"].to(self.device)
                image_ids = batch["image_id"]

                outputs = self.model(images)

                # Get Top-5 indices
                _, indices = torch.topk(outputs, k=Config.TOP_K, dim=1)
                indices = indices.cpu().numpy()

                # Decode indices to Hotel IDs and format submission
                for i, img_id in enumerate(image_ids):
                    pred_indices = indices[i]
                    pred_hotel_ids = label_encoder.inverse_transform(pred_indices)

                    # Space-delimited list of hotel IDs
                    pred_str = " ".join(map(str, pred_hotel_ids))
                    results.append({"image": img_id, "hotel_id": pred_str})

        # Save Submission
        submission_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(submission_path), exist_ok=True)
        submission_df.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

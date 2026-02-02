import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd

from library.config import Config
from library.utils import calculate_score, get_label_map, seed_everything
from library.network import AppleDiseaseModel
from library.data_loader import get_dataloaders


class Trainer:
    """
    Trainer class to handle model training, validation, and inference.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        num_epochs=Config.EPOCHS,
        patience=3,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_epochs = num_epochs
        self.patience = patience

        # Loss function for multi-label classification
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=num_epochs
        )

        # State tracking
        self.best_score = -1.0
        self.early_stop_counter = 0
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        # Mixed Precision Scaler
        self.scaler = torch.cuda.amp.GradScaler()

    def train_one_epoch(self):
        """
        Trains the model for one epoch using Mixed Precision.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, targets in self.train_loader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Mixed Precision Forward pass
            with torch.cuda.amp.autocast():
                logits = self.model(images)
                loss = self.criterion(logits, targets)

            # Mixed Precision Backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Accumulate loss
            batch_size = images.size(0)
            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate_one_epoch(self):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)
                targets = targets.to(self.device)

                with torch.cuda.amp.autocast():
                    logits = self.model(images)
                    loss = self.criterion(logits, targets)

                batch_size = images.size(0)
                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to convert logits to probabilities for metric calculation
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)

        # Calculate F1 Score
        # calculate_score handles thresholding internally (default 0.5)
        epoch_score = calculate_score(
            all_targets, all_preds, threshold=0.5, average="macro"
        )

        return epoch_loss, epoch_score

    def fit(self):
        """
        Runs the full training loop with early stopping.
        """
        print(f"Starting training on {self.device} for {self.num_epochs} epochs.")

        for epoch in range(self.num_epochs):
            start_time = time.time()

            train_loss = self.train_one_epoch()
            val_loss, val_score = self.validate_one_epoch()

            # Step the scheduler
            self.scheduler.step()

            duration = time.time() - start_time

            # Print metrics (Full precision as requested)
            print(
                f"Epoch {epoch + 1}/{self.num_epochs} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val F1: {val_score}"
            )

            # Checkpoint and Early Stopping
            if val_score > self.best_score:
                self.best_score = val_score
                self.early_stop_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with F1 Score: {val_score}")
            else:
                self.early_stop_counter += 1
                if self.early_stop_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs.")
                    break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Loading best model for inference...")
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
        else:
            print("Warning: Best model not found. Using current model weights.")

        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                logits = self.model(images)
                probs = torch.sigmoid(logits)
                all_preds.append(probs.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)

        # Convert probabilities to labels
        # Threshold = 0.5
        binary_preds = (all_preds > 0.5).astype(int)

        # Map indices to class names
        _, int2str = get_label_map()

        submission_rows = []
        # Access image IDs from the dataset dataframe
        image_ids = test_loader.dataset.df["image"].values

        for idx, binary_vector in enumerate(binary_preds):
            image_id = image_ids[idx]

            # Get list of indices where value is 1
            class_indices = np.where(binary_vector == 1)[0]

            if len(class_indices) == 0:
                # Fallback: if no class is predicted, pick the one with max probability
                # or default to 'healthy' depending on strategy.
                # Here we pick max probability to ensure at least one label.
                max_idx = np.argmax(all_preds[idx])
                label_str = int2str[max_idx]
            else:
                labels = [int2str[i] for i in class_indices]
                label_str = " ".join(labels)

            submission_rows.append({"image": image_id, "labels": label_str})

        # Create DataFrame and save
        submission_df = pd.DataFrame(submission_rows)
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def train_model(debug=Config.DEBUG):
    """
    Main function to setup and run training.
    """
    # 1. Setup
    Config.setup_reproducibility()
    device = torch.device(Config.DEVICE)

    # 2. Data
    train_loader, val_loader, test_loader = get_dataloaders(debug=debug)

    # 3. Model
    model = AppleDiseaseModel(pretrained=True)
    model.to(device)

    # 4. Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY,
        num_epochs=Config.EPOCHS,
    )

    # 5. Execute
    trainer.fit()

    # 6. Predict
    trainer.predict(test_loader)

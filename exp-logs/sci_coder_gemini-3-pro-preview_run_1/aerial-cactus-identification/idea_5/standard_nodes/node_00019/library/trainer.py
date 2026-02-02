import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import set_seed, mixup_data, mixup_criterion
from library.model import RepVGGCactus


class Trainer:
    """
    Trainer class for the Cactus Identification task using RepVGG.
    Handles training, validation, checkpointing, and submission generation.
    """

    def __init__(self, model, device=Config.DEVICE):
        self.model = model.to(device)
        self.device = device

        # Loss function for binary classification (accepts logits)
        self.criterion = nn.BCEWithLogitsLoss()

        # Optimizer: AdamW for better weight decay handling
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler: Cosine Annealing
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.ETA_MIN
        )

        # Performance Tracking
        self.best_auc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader):
        """
        Trains the model for one epoch using Mixup regularization.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for images, labels in train_loader:
            batch_size = images.size(0)
            images = images.to(self.device)
            # Ensure labels are (B, 1) floats
            labels = labels.to(self.device).view(-1, 1)

            # Apply Mixup Augmentation
            mixed_images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(mixed_images)

            # Calculate Mixup Loss
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        return running_loss / dataset_size

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                batch_size = images.size(0)
                images = images.to(self.device)
                labels = labels.to(self.device).view(-1, 1)

                # Forward pass (no Mixup)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Store probabilities and true labels for AUC
                probs = torch.sigmoid(outputs)
                all_preds.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        epoch_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        # Calculate ROC AUC
        try:
            epoch_auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            # Handle edge case where only one class is present in the batch/split
            epoch_auc = 0.5

        return epoch_loss, epoch_auc

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with early stopping and model checkpointing.
        """
        print(f"Starting training on {self.device} for {epochs} epochs.")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            # Step the scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | "
                f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                f"Val AUC: {val_auc}"
            )

            # Checkpoint and Early Stopping Logic
            if val_auc > self.best_auc:
                self.best_auc = val_auc
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
            else:
                self.patience_counter += 1

            if self.patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break

        print(f"Training finished. Best Validation AUC: {self.best_auc}")

    def predict(self, test_loader):
        """
        Performs inference on the test set.
        Loads the best model, applies structural re-parameterization, and uses TTA.
        """
        # 1. Load best weights
        if os.path.exists(Config.CHECKPOINT_PATH):
            state_dict = torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print("Loaded best model checkpoint for inference.")
        else:
            print("Warning: No checkpoint found. Using current model state.")

        # 2. Switch to evaluation mode
        self.model.eval()

        # 3. Reparameterize: Fuse multi-branch blocks into single convs
        # This optimizes inference speed and memory.
        print("Reparameterizing model (fusing blocks)...")
        self.model.reparameterize()

        all_preds = []

        # 4. Inference with Test Time Augmentation (TTA)
        print("Starting inference with TTA...")
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)

                # TTA 1: Original
                out1 = torch.sigmoid(self.model(images))

                if Config.USE_TTA:
                    # TTA 2: Horizontal Flip
                    out2 = torch.sigmoid(self.model(torch.flip(images, [3])))

                    # TTA 3: Vertical Flip
                    out3 = torch.sigmoid(self.model(torch.flip(images, [2])))

                    # TTA 4: Rotate 180 (H + V Flip)
                    out4 = torch.sigmoid(self.model(torch.flip(images, [2, 3])))

                    # Average predictions
                    batch_preds = (out1 + out2 + out3 + out4) / 4.0
                else:
                    batch_preds = out1

                all_preds.append(batch_preds.cpu().numpy())

        return np.concatenate(all_preds).flatten()

    def generate_submission(self, test_loader):
        """
        Generates the submission CSV file using predictions from the best model.
        """
        predictions = self.predict(test_loader)

        # Load test metadata to map predictions to IDs
        if not os.path.exists(Config.TEST_METADATA):
            raise FileNotFoundError(
                f"Test metadata not found at {Config.TEST_METADATA}"
            )

        df_test = pd.read_csv(Config.TEST_METADATA)

        # Safety check for alignment
        if len(predictions) != len(df_test):
            print(
                f"Error: Prediction count ({len(predictions)}) != Test set size ({len(df_test)})"
            )

        # Assign predictions
        df_test["has_cactus"] = predictions

        # Save submission
        submission_path = Config.SUBMISSION_PATH
        # Ensure only required columns are saved
        df_test[["id", "has_cactus"]].to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

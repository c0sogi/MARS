import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config
from library.utils import seed_everything, calculate_class_weights, compute_roc_auc
from library.model import WhaleEfficientNetV2


class Trainer:
    """
    Trainer class responsible for training and validating the Whale Call Detection model.
    """

    def __init__(self, train_loader, val_loader):
        """
        Initialize the Trainer.

        Args:
            train_loader (DataLoader): DataLoader for the training set.
            val_loader (DataLoader): DataLoader for the validation set.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        self.model = WhaleEfficientNetV2(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # Initialize Loss Function
        # Calculate class weights based on training data distribution
        train_df = pd.read_csv(Config.TRAIN_CSV)
        if Config.USE_WEIGHTED_LOSS:
            pos_weight = calculate_class_weights(train_df, target_col="label")
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(self.device))
        else:
            self.criterion = nn.BCEWithLogitsLoss()

        # Initialize Optimizer (AdamW)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Scheduler (Cosine Annealing)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Initialize Mixed Precision Scaler
        self.scaler = torch.amp.GradScaler("cuda")

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device).unsqueeze(1)  # Shape: (Batch, 1)
            batch_size = images.size(0)

            self.optimizer.zero_grad()

            # Mixup Augmentation Logic
            use_mixup = Config.USE_MIXUP and Config.MIXUP_ALPHA > 0

            if use_mixup:
                # Sample lambda from Beta distribution
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

                # Shuffle indices for mixing
                index = torch.randperm(batch_size).to(self.device)

                # Mix inputs
                mixed_images = lam * images + (1 - lam) * images[index]

                # Forward pass with Mixed Precision
                with torch.amp.autocast("cuda"):
                    outputs = self.model(mixed_images)

                    # Compute mixed loss: mix the weighted scalar losses of the input pair
                    loss_a = self.criterion(outputs, labels)
                    loss_b = self.criterion(outputs, labels[index])
                    loss = lam * loss_a + (1 - lam) * loss_b
            else:
                # Standard Forward pass
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

            # Backward pass and Optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).unsqueeze(1)
                batch_size = images.size(0)

                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                # Apply sigmoid to convert logits to probabilities
                probs = torch.sigmoid(outputs)

                all_preds.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        val_loss = running_loss / dataset_size

        # Concatenate all batches
        all_preds = np.concatenate(all_preds)
        all_targets = np.concatenate(all_targets)

        # Compute AUC
        val_auc = compute_roc_auc(all_targets, all_preds)

        return val_loss, val_auc

    def fit(self):
        """
        Main training loop. Handles epochs, scheduler stepping, and model checkpointing.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on {self.device}...")
        best_auc = 0.0
        best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        for epoch in range(Config.EPOCHS):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_auc = self.validate()

            # Step Scheduler
            self.scheduler.step()

            end_time = time.time()
            epoch_time = end_time - start_time

            # Print Metrics (Full Precision)
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss} | "
                f"Val Loss: {val_loss} | "
                f"Val AUC: {val_auc}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), best_model_path)
                print(f"Validation AUC improved. Saved best model to {best_model_path}")

        print(f"Training finished. Best Validation AUC: {best_auc}")
        return best_auc

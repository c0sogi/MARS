import os
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from library.config import Config
from library.utils import MetricMonitor, save_checkpoint, load_checkpoint
from library.losses import WeightedBCELoss, MixupLoss


class Trainer:
    def __init__(self, model, optimizer, scheduler=None, device=Config.DEVICE):
        """
        Initializes the Trainer.

        Args:
            model: The PyTorch model to train.
            optimizer: The optimizer.
            scheduler: Learning rate scheduler (optional).
            device: Device to run training on.
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        # Calculate positive class weight based on training metadata
        # to handle class imbalance (Inverse Class Frequency Weighting)
        train_df = pd.read_csv(Config.TRAIN_CSV)
        # Filter for valid labels just in case
        valid_train = train_df[train_df["label"].isin([0, 1])]
        neg_count = len(valid_train[valid_train["label"] == 0])
        pos_count = len(valid_train[valid_train["label"] == 1])

        # Weight = N_neg / N_pos
        pos_weight_val = neg_count / pos_count if pos_count > 0 else 1.0

        # Initialize Loss Functions
        self.criterion = WeightedBCELoss(pos_weight_value=pos_weight_val, device=device)
        self.mixup_criterion = MixupLoss(self.criterion)

    def train_one_epoch(self, train_loader):
        """
        Trains the model for one epoch using Mixup augmentation.
        """
        self.model.train()
        metric_monitor = MetricMonitor()

        for batch in train_loader:
            images, targets, _ = batch
            images = images.to(self.device)
            targets = targets.to(self.device)

            batch_size = images.size(0)

            # Apply Mixup
            # Sample lambda from Beta distribution
            lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)

            # Generate random permutation for mixing
            index = torch.randperm(batch_size).to(self.device)

            mixed_images = lam * images + (1 - lam) * images[index]
            target_a, target_b = targets, targets[index]

            # Forward pass
            outputs = self.model(mixed_images)

            # Calculate Mixup Loss
            loss = self.mixup_criterion(outputs, target_a, target_b, lam)

            # Backward pass and Optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            metric_monitor.update("Loss", loss.item(), batch_size)

        return metric_monitor.metrics["Loss"]["avg"]

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss and AUC.
        """
        self.model.eval()
        metric_monitor = MetricMonitor()
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for batch in val_loader:
                images, targets, _ = batch
                images = images.to(self.device)
                targets = targets.to(self.device)

                # Forward pass (No Mixup)
                outputs = self.model(images)

                # Calculate Standard Loss
                loss = self.criterion(outputs, targets)
                metric_monitor.update("Loss", loss.item(), images.size(0))

                # Store predictions for AUC calculation
                probs = torch.sigmoid(outputs).cpu().numpy()
                targets_np = targets.cpu().numpy()

                preds_list.extend(probs.flatten())
                targets_list.extend(targets_np.flatten())

        # Calculate AUC
        try:
            auc = roc_auc_score(targets_list, preds_list)
        except ValueError:
            # Handle edge case where only one class is present in validation batch
            auc = 0.5

        return metric_monitor.metrics["Loss"]["avg"], auc

    def fit(self, train_loader, val_loader, epochs, checkpoint_name="best_model.pth"):
        """
        Main training loop. Handles epochs, logging, and checkpointing.
        """
        best_auc = 0.0
        save_path = os.path.join(Config.WORKING_DIR, checkpoint_name)

        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            if self.scheduler:
                self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch: {epoch} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val AUC: {val_auc}"
            )

            # Save best model
            if val_auc > best_auc:
                best_auc = val_auc
                save_checkpoint(self.model, self.optimizer, epoch, val_auc, save_path)

        # Reload the best model weights for future use (inference/pseudo-labeling)
        print(f"Reloading best model from {save_path} with AUC {best_auc}")
        load_checkpoint(self.model, save_path, device=self.device)

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        Returns a dictionary mapping clip names to probabilities.
        """
        self.model.eval()
        results = {}

        with torch.no_grad():
            for batch in test_loader:
                images, _, clips = batch
                images = images.to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()

                for clip, prob in zip(clips, probs):
                    results[clip] = prob

        return results

    def generate_submission(self, test_loader, output_file=Config.SUBMISSION_FILE):
        """
        Generates predictions and saves them to a CSV file in the submission format.
        """
        print("Generating predictions for submission...")
        predictions = self.predict(test_loader)

        # Create DataFrame from results
        df = pd.DataFrame(list(predictions.items()), columns=["clip", "probability"])

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # Save to CSV
        df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")

        return predictions

import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from library.config import Config
from library.utils import (
    AverageMeter,
    calculate_auc,
    print_training_status,
    print_validation_metric,
    seed_everything,
)
from library.data_loader import get_dataloaders
from library.model import GroupedEfficientNet


class Trainer:
    def __init__(self, device=Config.DEVICE):
        self.device = device
        self.model = GroupedEfficientNet().to(self.device)

        # Optimizer with aggressive weight decay as per idea
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Loss functions
        self.criterion = nn.BCEWithLogitsLoss()

        self.best_auc = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, epoch, total_epochs):
        self.model.train()
        loss_meter = AverageMeter("Loss")

        total_batches = len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device).unsqueeze(1)  # (B, 1)

            self.optimizer.zero_grad()

            # Standard Forward Pass
            logits = self.model(inputs)
            loss = self.criterion(logits, targets)

            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), inputs.size(0))

            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == total_batches:
                print_training_status(
                    epoch, total_epochs, batch_idx + 1, total_batches, loss_meter
                )

        return loss_meter.avg

    def validate_one_epoch(self, val_loader):
        self.model.eval()
        loss_meter = AverageMeter("Val Loss")

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for view_a, targets in val_loader:
                view_a = view_a.to(self.device)
                targets = targets.to(self.device).unsqueeze(1)

                logits = self.model(view_a)
                loss = self.criterion_bce(logits, targets)

                probs = torch.sigmoid(logits)

                loss_meter.update(loss.item(), view_a.size(0))

                all_targets.extend(targets.cpu().numpy().flatten())
                all_preds.extend(probs.cpu().numpy().flatten())

        auc = calculate_auc(all_targets, all_preds)
        return loss_meter.avg, auc

    def run_training(self, debug=False):
        seed_everything(Config.SEED)

        # Load Metadata
        train_df = pd.read_csv(Config.TRAIN_METADATA)
        val_df = pd.read_csv(Config.VAL_METADATA)
        test_df = pd.read_csv(Config.TEST_METADATA)

        # Get Dataloaders
        train_loader, val_loader, test_loader = get_dataloaders(
            train_df, val_df, test_df, debug=debug
        )

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch, Config.NUM_EPOCHS)

            # Validate
            val_loss, val_auc = self.validate_one_epoch(val_loader)

            print_validation_metric("Loss", val_loss)
            print_validation_metric("AUC", val_auc)

            # Checkpoint & Early Stopping
            if val_auc > self.best_auc:
                print(
                    f"AUC improved from {self.best_auc} to {val_auc}. Saving model..."
                )
                self.best_auc = val_auc
                torch.save(self.model.state_dict(), Config.CHECKPOINT_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )

            if self.patience_counter >= Config.PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation AUC: {self.best_auc}")

        # Run Inference
        self.predict_and_submit(test_loader)

    def predict_and_submit(self, test_loader):
        print("Starting inference on test set...")

        # Load best model
        if os.path.exists(Config.CHECKPOINT_PATH):
            self.model.load_state_dict(
                torch.load(Config.CHECKPOINT_PATH, map_location=self.device)
            )
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        predictions = []
        ids = []

        with torch.no_grad():
            for tta_tensor, subject_ids in test_loader:
                # tta_tensor shape: (Batch, 3, 12, H, W)
                # subject_ids: tuple/list of IDs

                batch_size, num_views, c, h, w = tta_tensor.shape

                # Flatten to (Batch * 3, 12, H, W) for efficient batch processing
                flat_input = tta_tensor.view(-1, c, h, w).to(self.device)

                # Forward pass
                flat_logits = self.model(flat_input)
                flat_probs = torch.sigmoid(flat_logits)  # (Batch * 3, 1)

                # Reshape back to (Batch, 3)
                probs = flat_probs.view(batch_size, num_views)

                # Average over the TTA views
                avg_probs = probs.mean(dim=1).cpu().numpy()

                predictions.extend(avg_probs)
                ids.extend(subject_ids.numpy())

        # Create Submission DataFrame
        submission_df = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": predictions})

        # Format BraTS21ID as 5-digit string for consistency with sample (though sample uses int in prompt logic, usually competition requires string or int matching sample)
        # The prompt sample_submission.csv shows BraTS21ID as int (e.g. 356, 140).
        # We will sort by ID to be tidy.
        submission_df = submission_df.sort_values("BraTS21ID")

        # Save
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

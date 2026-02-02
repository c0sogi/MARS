import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import (
    set_seed,
    mixup_data,
    mixup_criterion,
    average_weights,
    update_bn,
)
from library.data import get_dataloaders
from library.model import MetadataGatedRepVGG


class Trainer:
    """
    Manages the training, validation, SWA, and inference processes for the Cactus Classifier.
    """

    def __init__(self):
        # Reproducibility
        set_seed(Config.SEED)
        self.device = Config.DEVICE

        # Data Loading (Cached)
        print("Initializing Data Loaders...")
        self.train_loader, self.val_loader, self.test_loader, self.test_ids = (
            get_dataloaders(load_cached_data=True)
        )

        # Model Initialization
        print(f"Initializing Model: {Config.MODEL_NAME}")
        self.model = MetadataGatedRepVGG().to(self.device)

        # Optimization Setup
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Paths
        self.working_dir = Config.CACHE_DIR
        self.checkpoint_dir = os.path.join(self.working_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch, scheduler=None, is_swa=False):
        """
        Runs one epoch of training with Mixup.
        """
        self.model.train()
        running_loss = 0.0

        for batch_idx, (images, metadata, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            metadata = metadata.to(self.device)
            labels = labels.to(self.device)

            # Apply Mixup
            images, targets_a, targets_b, lam = mixup_data(
                images, labels, Config.MIXUP_ALPHA, self.device
            )

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(images, metadata)

            # Mixup Loss
            loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            loss.backward()
            self.optimizer.step()

            # Step scheduler per batch if in SWA cyclic mode
            if scheduler and is_swa:
                scheduler.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    def validate(self):
        """
        Evaluates the model on the validation set and computes ROC AUC.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for images, metadata, labels in self.val_loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)

                # Forward pass
                outputs = self.model(images, metadata)

                # Get probabilities for class 1 (cactus)
                probs = F.softmax(outputs, dim=1)[:, 1]

                all_preds.extend(probs.cpu().numpy())
                all_targets.extend(labels.numpy())

        try:
            auc = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc = 0.5

        return auc

    def run(self):
        """
        Executes the full training pipeline: Convergence -> SWA Exploration -> Averaging.
        """
        print("Starting Training Pipeline...")

        # --- Phase 1: Convergence ---
        print(f"Phase 1: Convergence ({Config.EPOCHS_CONVERGENCE} epochs)")

        # Standard Scheduler for convergence
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS_CONVERGENCE
        )

        best_auc = 0.0
        best_model_path = os.path.join(
            self.checkpoint_dir, "best_model_convergence.pth"
        )

        for epoch in range(Config.EPOCHS_CONVERGENCE):
            loss = self.train_epoch(epoch)
            scheduler.step()
            auc = self.validate()

            print(
                f"Epoch {epoch+1}/{Config.TOTAL_EPOCHS} (Convergence) - Loss: {loss:.6f}, Val AUC: {auc}"
            )

            if auc > best_auc:
                best_auc = auc
                torch.save(self.model.state_dict(), best_model_path)

        print(f"Convergence Phase Complete. Best AUC: {best_auc}")

        # --- Phase 2: SWA Exploration ---
        print(f"Phase 2: SWA Exploration ({Config.EPOCHS_SWA} epochs)")

        swa_checkpoints = []

        # Reset Learning Rate for SWA
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = Config.SWA_LR

        # Steps per epoch for cyclic scheduler
        steps_per_epoch = len(self.train_loader)

        for i in range(Config.EPOCHS_SWA):
            current_epoch = Config.EPOCHS_CONVERGENCE + i

            # Cyclic Cosine Scheduler (restarts every epoch)
            swa_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=steps_per_epoch, eta_min=1e-6
            )

            loss = self.train_epoch(current_epoch, scheduler=swa_scheduler, is_swa=True)
            auc = self.validate()

            print(
                f"Epoch {current_epoch+1}/{Config.TOTAL_EPOCHS} (SWA) - Loss: {loss:.6f}, Val AUC: {auc}"
            )

            # Save SWA Snapshot
            ckpt_path = os.path.join(self.checkpoint_dir, f"swa_snapshot_{i}.pth")
            torch.save(self.model.state_dict(), ckpt_path)
            swa_checkpoints.append(ckpt_path)

        # --- Phase 3: Averaging ---
        print("Phase 3: Weight Averaging")

        # Average weights
        avg_state_dict = average_weights(swa_checkpoints)
        self.model.load_state_dict(avg_state_dict)

        # Update Batch Normalization Statistics
        print("Updating BN statistics...")
        update_bn(self.train_loader, self.model, self.device)

        # Validate Final SWA Model
        final_auc = self.validate()
        print(f"Final SWA Model Val AUC: {final_auc}")

        # Save Final Model
        final_model_path = os.path.join(self.working_dir, "final_swa_model.pth")
        torch.save(self.model.state_dict(), final_model_path)
        print(f"Saved final model to {final_model_path}")

        return final_model_path

    def predict(self, model_path):
        """
        Performs inference using TTA and generates the submission file.
        """
        print("Starting Inference with TTA...")

        # Load Model
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        # Switch to Deploy Mode (Structural Re-parameterization)
        print("Switching model to deploy mode (fusing RepVGG blocks)...")
        self.model.switch_to_deploy()
        self.model.eval()
        self.model.to(self.device)

        predictions = []

        with torch.no_grad():
            for images, metadata in self.test_loader:
                images = images.to(self.device)
                metadata = metadata.to(self.device)

                # --- Test Time Augmentation (4 Views) ---

                # View 1: Original
                out1 = F.softmax(self.model(images, metadata), dim=1)[:, 1]

                # View 2: Horizontal Flip
                out2 = F.softmax(self.model(images.flip(3), metadata), dim=1)[:, 1]

                # View 3: Vertical Flip
                out3 = F.softmax(self.model(images.flip(2), metadata), dim=1)[:, 1]

                # View 4: Rotate 180 (equivalent to H-Flip + V-Flip)
                out4 = F.softmax(self.model(images.flip(2).flip(3), metadata), dim=1)[
                    :, 1
                ]

                # Average Predictions
                avg_probs = (out1 + out2 + out3 + out4) / 4.0
                predictions.extend(avg_probs.cpu().numpy())

        # Generate Submission CSV
        df = pd.DataFrame({"id": self.test_ids, "has_cactus": predictions})

        df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def main():
    trainer = Trainer()
    final_model_path = trainer.run()
    trainer.predict(final_model_path)

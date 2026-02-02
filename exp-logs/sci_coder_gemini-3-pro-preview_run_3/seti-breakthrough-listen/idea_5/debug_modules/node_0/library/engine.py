import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from library.utils import Config, set_seed
from library.model import SiameseSpatialFusionNet
from library.data import TechnosignatureDataset


class ModelEngine:
    def __init__(self):
        set_seed(Config.SEED)
        Config.setup()

        # device
        self.device = Config.DEVICE

        # Model
        self.model = SiameseSpatialFusionNet().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS
        )

        # Loss
        self.criterion = nn.BCEWithLogitsLoss()

        # Paths
        self.best_model_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        self.submission_dir = "./submission"
        os.makedirs(self.submission_dir, exist_ok=True)

    def train_one_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        dataset_size = 0

        for batch_idx, (images, targets) in enumerate(train_loader):
            img_on, img_off = images
            img_on = img_on.to(self.device)
            img_off = img_off.to(self.device)
            targets = targets.to(self.device).view(-1, 1)

            batch_size = img_on.size(0)

            # Mixup
            if Config.MIXUP_ALPHA > 0:
                lam = np.random.beta(Config.MIXUP_ALPHA, Config.MIXUP_ALPHA)
                index = torch.randperm(batch_size).to(self.device)

                img_on_mixed = lam * img_on + (1 - lam) * img_on[index]
                img_off_mixed = lam * img_off + (1 - lam) * img_off[index]
                targets_mixed = lam * targets + (1 - lam) * targets[index]

                outputs = self.model(img_on_mixed, img_off_mixed)
                loss = self.criterion(outputs, targets_mixed)
            else:
                outputs = self.model(img_on, img_off)
                loss = self.criterion(outputs, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * batch_size
            dataset_size += batch_size

        epoch_loss = running_loss / dataset_size
        return epoch_loss

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        dataset_size = 0

        all_targets = []
        all_preds = []

        with torch.no_grad():
            for images, targets in val_loader:
                img_on, img_off = images
                img_on = img_on.to(self.device)
                img_off = img_off.to(self.device)
                targets = targets.to(self.device).view(-1, 1)

                batch_size = img_on.size(0)

                outputs = self.model(img_on, img_off)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * batch_size
                dataset_size += batch_size

                probs = torch.sigmoid(outputs)
                all_targets.extend(targets.cpu().numpy())
                all_preds.extend(probs.cpu().numpy())

        epoch_loss = running_loss / dataset_size
        all_targets = np.array(all_targets)
        all_preds = np.array(all_preds)

        try:
            auc_score = roc_auc_score(all_targets, all_preds)
        except ValueError:
            auc_score = 0.5

        return epoch_loss, auc_score

    def predict_with_tta(self, test_loader):
        """
        Predict on test set using Test Time Augmentation.
        TTA: Original, Horizontal Flip, Vertical Flip.
        """
        self.model.eval()
        predictions = []
        ids = []

        # We need IDs from the dataset. The loader returns (images, targets),
        # but the dataset has the dataframe. We'll iterate the loader and
        # assume order is preserved, but to be safe we can access the dataset directly
        # or rely on the sequential sampler. The provided dataset __getitem__
        # doesn't return IDs. We will rely on the order of the test.csv metadata.

        with torch.no_grad():
            for images, _ in test_loader:
                img_on, img_off = images
                img_on = img_on.to(self.device)
                img_off = img_off.to(self.device)

                # 1. Original
                out1 = self.model(img_on, img_off)
                prob1 = torch.sigmoid(out1)

                # 2. Horizontal Flip (dim 3 for W)
                img_on_h = torch.flip(img_on, [3])
                img_off_h = torch.flip(img_off, [3])
                out2 = self.model(img_on_h, img_off_h)
                prob2 = torch.sigmoid(out2)

                # 3. Vertical Flip (dim 2 for H)
                img_on_v = torch.flip(img_on, [2])
                img_off_v = torch.flip(img_off, [2])
                out3 = self.model(img_on_v, img_off_v)
                prob3 = torch.sigmoid(out3)

                # Average
                avg_prob = (prob1 + prob2 + prob3) / 3.0
                predictions.extend(avg_prob.cpu().numpy().flatten())

        return predictions

    def run(self, patience=3):
        # Data Loaders
        train_ds = TechnosignatureDataset(
            os.path.join(Config.METADATA_DIR, "train.csv"), data_type="train"
        )
        val_ds = TechnosignatureDataset(
            os.path.join(Config.METADATA_DIR, "val.csv"), data_type="val"
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        best_auc = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {Config.NUM_EPOCHS} epochs...")

        for epoch in range(Config.NUM_EPOCHS):
            start_time = time.time()

            train_loss = self.train_one_epoch(train_loader)
            val_loss, val_auc = self.validate(val_loader)

            self.scheduler.step()

            end_time = time.time()
            duration = end_time - start_time

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Time: {duration:.2f}s | "
                f"Train Loss: {train_loss:.8f} | "
                f"Val Loss: {val_loss:.8f} | "
                f"Val AUC: {val_auc:.15f}"
            )

            # Checkpoint
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(self.model.state_dict(), self.best_model_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training complete. Best Val AUC: {best_auc:.15f}")

        # --- Inference ---
        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.best_model_path, map_location=self.device)
        )

        test_metadata_path = os.path.join(Config.METADATA_DIR, "test.csv")
        test_ds = TechnosignatureDataset(test_metadata_path, data_type="test")
        test_loader = DataLoader(
            test_ds,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print("Generating predictions with TTA...")
        preds = self.predict_with_tta(test_loader)

        # Create submission dataframe
        df_test = pd.read_csv(test_metadata_path)
        df_test["target"] = preds

        # Keep only required columns
        submission_df = df_test[["id", "target"]]

        save_path = os.path.join(self.submission_dir, "submission.csv")
        submission_df.to_csv(save_path, index=False)
        print(f"Submission saved to {save_path}")


def main():
    engine = ModelEngine()
    engine.run()

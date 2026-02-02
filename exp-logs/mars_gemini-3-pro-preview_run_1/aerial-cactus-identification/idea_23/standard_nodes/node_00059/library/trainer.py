import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from library.config import Config
from library.utils import seed_everything, mixup_data, mixup_criterion, MetricMonitor
from library.dataset import CactusDataset
from library.model import QualityRepVGG


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.num_epochs = Config.EPOCHS
        self.batch_size = Config.BATCH_SIZE
        self.n_folds = Config.N_FOLDS

        # Loss functions
        # BCEWithLogitsLoss for classification
        self.criterion_cls = nn.BCEWithLogitsLoss()
        # MSELoss for quality regression (Auxiliary task)
        self.criterion_qual = nn.MSELoss()

    def _get_dataloaders(self, fold_idx):
        """
        Creates train and validation dataloaders for a specific fold.
        Combines provided train/val metadata and splits them using StratifiedKFold
        to ensure robust cross-validation as per the strategy.
        """
        # Load metadata
        full_df = pd.read_csv(Config.TRAIN_META_PATH)

        # Synchronize data filtering logic with Dataset class
        if Config.DEBUG:
            full_df = full_df.head(100)

        # Create temporary metadata file for the full dataset to pass to CactusDataset
        full_meta_path = os.path.join(Config.WORK_DIR, "full_train_metadata.csv")
        full_df.to_csv(full_meta_path, index=False)

        # Instantiate the full dataset
        # We need two instances: one for training (with augs) and one for validation (no augs)
        full_dataset_train = CactusDataset(
            metadata_path=full_meta_path, mode="train", load_cached_data=True
        )
        full_dataset_val = CactusDataset(
            metadata_path=full_meta_path, mode="val", load_cached_data=True
        )

        # Split indices
        skf = StratifiedKFold(
            n_splits=self.n_folds, shuffle=True, random_state=Config.SEED
        )

        # We need targets for stratification
        targets = full_df["has_cactus"].values

        # Get indices for the current fold
        fold_splits = list(skf.split(np.zeros(len(targets)), targets))
        train_idx, val_idx = fold_splits[fold_idx]

        # Create Subsets
        train_subset = Subset(full_dataset_train, train_idx)
        val_subset = Subset(full_dataset_val, val_idx)

        # Create Loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader

    def train_epoch(self, train_loader, model, optimizer, scheduler, epoch):
        model.train()
        metric_monitor = MetricMonitor(float_precision=4)

        for batch_idx, (images, labels, qualities) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            qualities = qualities.to(self.device)

            # Apply Mixup to both images and targets (class + quality)
            mixed_images, labels_a, labels_b, qual_a, qual_b, lam = mixup_data(
                images, labels, qualities, alpha=Config.MIXUP_ALPHA, device=self.device
            )

            # Forward pass
            # QualityRepVGG returns (cls_out, qual_out) in training mode
            cls_preds, qual_preds = model(mixed_images)

            # Compute Multi-Task Loss
            loss = mixup_criterion(
                self.criterion_cls,
                self.criterion_qual,
                cls_preds,
                qual_preds,
                labels_a,
                labels_b,
                qual_a,
                qual_b,
                lam,
                Config.AUX_LOSS_WEIGHT,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update metrics
            metric_monitor.update("Loss", loss.item())

        return metric_monitor.avg["Loss"]

    def validate(self, val_loader, model):
        model.eval()
        metric_monitor = MetricMonitor(float_precision=None)

        preds_list = []
        targets_list = []

        with torch.no_grad():
            for images, labels, qualities in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                qualities = qualities.to(self.device)

                # Forward pass
                # Model returns tuple (cls, qual) if not deployed and aux head exists
                outputs = model(images)
                if isinstance(outputs, tuple):
                    cls_preds, qual_preds = outputs
                else:
                    cls_preds = outputs
                    qual_preds = None

                # Classification Loss
                loss_cls = self.criterion_cls(cls_preds, labels)
                metric_monitor.update("Loss_Cls", loss_cls.item())

                # Quality Loss (if available)
                if qual_preds is not None:
                    loss_qual = self.criterion_qual(qual_preds, qualities)
                    metric_monitor.update("Loss_Qual", loss_qual.item())

                # Store predictions for AUC
                probs = torch.sigmoid(cls_preds)
                preds_list.extend(probs.cpu().numpy())
                targets_list.extend(labels.cpu().numpy())

        # Calculate AUC
        try:
            auc = roc_auc_score(targets_list, preds_list)
        except ValueError:
            auc = 0.5

        metric_monitor.update("AUC", auc)
        return metric_monitor.avg, auc

    def run_fold(self, fold_idx):
        print(f"\n=== Starting Fold {fold_idx} ===")
        seed_everything(Config.SEED + fold_idx)

        # Data
        train_loader, val_loader = self._get_dataloaders(fold_idx)

        # Model
        model = QualityRepVGG(
            num_classes=Config.NUM_CLASSES,
            width_multiplier=Config.WIDTH_MULTIPLIER,
            deploy=False,
        )
        model = model.to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
        )

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.T_MAX, eta_min=Config.MIN_LR
        )

        # SWA Setup
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(optimizer, swa_lr=Config.SWA_LR)
        swa_start = Config.SWA_START_EPOCH

        best_auc = 0.0
        best_model_path = os.path.join(
            Config.CHECKPOINT_DIR, f"best_fold{fold_idx}.pth"
        )
        swa_model_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_fold{fold_idx}.pth")

        for epoch in range(1, self.num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(
                train_loader, model, optimizer, scheduler, epoch
            )

            # SWA Update Logic
            if Config.USE_SWA and epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            else:
                scheduler.step()

            # Validate
            # We validate the current model to track progress and save 'best'
            val_metrics, val_auc = self.validate(val_loader, model)

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{self.num_epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss Cls: {val_metrics['Loss_Cls']:.4f} | "
                f"Val AUC: {val_auc:.6f}"
            )

            # Save Best Model
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)

        # Finalize SWA
        if Config.USE_SWA:
            print("Finalizing SWA model...")
            # Update BN statistics for the SWA model using training data
            update_bn(train_loader, swa_model, device=self.device)
            # Save SWA model
            torch.save(swa_model.module.state_dict(), swa_model_path)
            print(f"SWA model saved to {swa_model_path}")

            # Validate SWA model
            _, val_auc_swa = self.validate(val_loader, swa_model.module)
            print(f"SWA Model Validation AUC: {val_auc_swa:.6f}")

            # Return SWA AUC as the final metric for this fold
            return val_auc_swa

        return best_auc

    def run_training(self):
        """
        Runs the full training pipeline for all folds.
        """
        fold_scores = []
        for fold in range(self.n_folds):
            auc = self.run_fold(fold)
            fold_scores.append(auc)

        print("\n=== Training Complete ===")
        print(f"Fold AUCs: {fold_scores}")
        print(f"Mean AUC: {np.mean(fold_scores):.6f}")

    def predict_test_set(self):
        print("\n=== Starting Inference on Test Set ===")

        # Load Test Data
        test_dataset = CactusDataset(
            metadata_path=Config.TEST_META_PATH, mode="test", load_cached_data=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Prepare to store predictions: (N_samples, N_folds)
        fold_preds = np.zeros((len(test_dataset), self.n_folds))

        for fold in range(self.n_folds):
            print(f"Predicting with Fold {fold} SWA model...")

            # Initialize model
            model = QualityRepVGG(
                num_classes=Config.NUM_CLASSES,
                width_multiplier=Config.WIDTH_MULTIPLIER,
                deploy=False,
            )

            # Load Weights (Prefer SWA, fallback to best)
            swa_path = os.path.join(Config.CHECKPOINT_DIR, f"swa_fold{fold}.pth")
            best_path = os.path.join(Config.CHECKPOINT_DIR, f"best_fold{fold}.pth")

            if os.path.exists(swa_path):
                weights = torch.load(swa_path, map_location=self.device)
            else:
                print(
                    f"SWA model not found, falling back to best model for fold {fold}"
                )
                weights = torch.load(best_path, map_location=self.device)

            model.load_state_dict(weights)

            # Reparameterize for inference (Fuses blocks, removes aux head)
            model.eval()
            model.reparameterize()
            model = model.to(self.device)

            # Prediction Loop with Test Time Augmentation (TTA)
            fold_probs = []

            with torch.no_grad():
                for images, _, _ in test_loader:
                    images = images.to(self.device)

                    # TTA: Original, HFlip, VFlip, HFlip+VFlip (Rot180)
                    # 1. Original
                    out1 = torch.sigmoid(model(images))

                    # 2. HFlip
                    out2 = torch.sigmoid(model(torch.flip(images, [3])))

                    # 3. VFlip
                    out3 = torch.sigmoid(model(torch.flip(images, [2])))

                    # 4. Rot180 (H+V)
                    out4 = torch.sigmoid(model(torch.flip(images, [2, 3])))

                    # Average TTA
                    avg_out = (out1 + out2 + out3 + out4) / 4.0
                    fold_probs.extend(avg_out.cpu().numpy().flatten())

            fold_preds[:, fold] = fold_probs

        # Average across folds
        final_preds = fold_preds.mean(axis=1)

        # Save Submission
        test_df = pd.read_csv(Config.TEST_META_PATH)
        if Config.DEBUG:
            test_df = test_df.head(100)
        test_df["has_cactus"] = final_preds

        # Ensure correct format
        submission_df = test_df[["id", "has_cactus"]]
        submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {Config.SUBMISSION_PATH}")

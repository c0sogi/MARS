import os
import time
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter

from library.config import Config
from library.utils import set_seed
from library.dataset import (
    PatchDetectorDataset,
    CharacterCropDataset,
    get_transforms,
    get_class_map,
)
from library.models import CenterNetDetector, ResNetClassifier
from library.losses import ModifiedFocalLoss, RegL1Loss


class DetectorTrainer:
    """
    Trainer for the CenterNet Detector (Stage 1).
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.output_dir = Config.MODEL_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Load Metadata
        self.train_df = pd.read_csv(Config.TRAIN_METADATA_PATH, keep_default_na=False)
        self.val_df = pd.read_csv(Config.VAL_METADATA_PATH, keep_default_na=False)

        if Config.DEBUG:
            print(
                f"Debug mode: Subsampling detector data to {Config.DEBUG_SAMPLE_SIZE}"
            )
            self.train_df = self.train_df.iloc[: Config.DEBUG_SAMPLE_SIZE]
            self.val_df = self.val_df.iloc[: Config.DEBUG_SAMPLE_SIZE]

        # Datasets
        self.train_dataset = PatchDetectorDataset(
            self.train_df,
            mode="train_detector",
            transform=get_transforms(
                mode="train_detector", img_size=Config.DETECTOR_INPUT_SIZE
            ),
        )
        self.val_dataset = PatchDetectorDataset(
            self.val_df,
            mode="val_detector",
            transform=get_transforms(
                mode="val_detector", img_size=Config.DETECTOR_INPUT_SIZE
            ),
        )

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.DETECTOR_BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.DETECTOR_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        self.model = CenterNetDetector(pretrained=True).to(self.device)

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.DETECTOR_LR)

        # Losses
        self.hm_loss_fn = ModifiedFocalLoss()
        self.reg_loss_fn = RegL1Loss()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        running_hm_loss = 0.0
        running_wh_loss = 0.0
        running_reg_loss = 0.0

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device)

            # Unpack targets
            target_hm = targets["hm"].to(self.device)
            target_wh = targets["wh"].to(self.device)
            target_reg = targets["reg"].to(self.device)
            target_reg_mask = targets["reg_mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward
            pred_hm, pred_wh, pred_reg = self.model(images)

            # Compute Losses
            loss_hm = self.hm_loss_fn(pred_hm, target_hm)
            loss_wh = self.reg_loss_fn(pred_wh, target_wh, target_reg_mask)
            loss_reg = self.reg_loss_fn(pred_reg, target_reg, target_reg_mask)

            # Weighted Sum (Standard CenterNet weights: 1.0, 0.1, 1.0)
            loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            running_hm_loss += loss_hm.item()
            running_wh_loss += loss_wh.item()
            running_reg_loss += loss_reg.item()

        n_batches = len(self.train_loader)
        return {
            "loss": running_loss / n_batches,
            "hm_loss": running_hm_loss / n_batches,
            "wh_loss": running_wh_loss / n_batches,
            "reg_loss": running_reg_loss / n_batches,
        }

    def validate(self):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for images, targets in self.val_loader:
                images = images.to(self.device)

                target_hm = targets["hm"].to(self.device)
                target_wh = targets["wh"].to(self.device)
                target_reg = targets["reg"].to(self.device)
                target_reg_mask = targets["reg_mask"].to(self.device)

                pred_hm, pred_wh, pred_reg = self.model(images)

                loss_hm = self.hm_loss_fn(pred_hm, target_hm)
                loss_wh = self.reg_loss_fn(pred_wh, target_wh, target_reg_mask)
                loss_reg = self.reg_loss_fn(pred_reg, target_reg, target_reg_mask)

                loss = loss_hm + 0.1 * loss_wh + 1.0 * loss_reg
                running_loss += loss.item()

        return running_loss / len(self.val_loader)

    def fit(self):
        print("Starting Detector Training...")
        best_val_loss = float("inf")

        epochs = Config.DETECTOR_EPOCHS
        if Config.DEBUG:
            epochs = 2

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            val_loss = self.validate()

            duration = time.time() - start_time

            print(f"Epoch {epoch}/{epochs} | Time: {duration:.2f}s")
            print(
                f"  Train Loss: {train_metrics['loss']:.6f} (HM: {train_metrics['hm_loss']:.6f}, WH: {train_metrics['wh_loss']:.6f}, Off: {train_metrics['reg_loss']:.6f})"
            )
            print(f"  Val Loss:   {val_loss:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.DETECTOR_CHECKPOINT)
                print(f"  -> Saved Best Detector Model (Loss: {best_val_loss:.6f})")

        print("Detector Training Completed.")


class ClassifierTrainer:
    """
    Trainer for the ResNet Classifier (Stage 2).
    """

    def __init__(self):
        self.device = Config.DEVICE
        self.output_dir = Config.MODEL_DIR
        os.makedirs(self.output_dir, exist_ok=True)

        # Load Metadata to get Class Map
        self.train_df_meta = pd.read_csv(
            Config.TRAIN_METADATA_PATH, keep_default_na=False
        )
        self.class_map = get_class_map(self.train_df_meta)
        self.num_classes = len(self.class_map)
        print(f"Classifier initialized for {self.num_classes} classes.")

        # Datasets
        # Note: CharacterCropDataset handles caching and parsing internally
        self.train_dataset = CharacterCropDataset(
            self.train_df_meta,
            self.class_map,
            split_name="train",
            mode="train_classifier",
            transform=get_transforms(
                mode="train_classifier", img_size=Config.CLASSIFIER_INPUT_SIZE
            ),
        )

        val_df_meta = pd.read_csv(Config.VAL_METADATA_PATH, keep_default_na=False)
        self.val_dataset = CharacterCropDataset(
            val_df_meta,
            self.class_map,
            split_name="val",
            mode="val_classifier",
            transform=get_transforms(
                mode="val_classifier", img_size=Config.CLASSIFIER_INPUT_SIZE
            ),
        )

        if Config.DEBUG:
            print(
                f"Debug mode: Subsampling classifier data to {Config.DEBUG_SAMPLE_SIZE}"
            )
            self.train_dataset.samples = self.train_dataset.samples[
                : Config.DEBUG_SAMPLE_SIZE
            ]
            self.val_dataset.samples = self.val_dataset.samples[
                : Config.DEBUG_SAMPLE_SIZE
            ]

        # Class Imbalance Handling (WeightedRandomSampler)
        # We need to compute weights for the sampler
        if not Config.DEBUG:
            print("Computing class weights for balanced sampling...")
            targets = [s["class_id"] for s in self.train_dataset.samples]
            class_counts = Counter(targets)

            # Weight = 1 / count
            weights = []
            for t in targets:
                count = class_counts.get(t, 0)
                if count > 0:
                    weights.append(1.0 / count)
                else:
                    weights.append(0)

            sampler = WeightedRandomSampler(
                weights, num_samples=len(weights), replacement=True
            )
            shuffle = False
            print("Sampler created.")
        else:
            sampler = None
            shuffle = True

        # Dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=Config.CLASSIFIER_BATCH_SIZE,
            sampler=sampler,
            shuffle=shuffle,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=Config.CLASSIFIER_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Model
        self.model = ResNetClassifier(num_classes=self.num_classes, pretrained=True).to(
            self.device
        )

        # Optimizer & Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.CLASSIFIER_LR)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        n_batches = len(self.train_loader)
        return {
            "loss": running_loss / n_batches,
            "acc": correct / total if total > 0 else 0.0,
        }

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        n_batches = len(self.val_loader)
        return {
            "loss": running_loss / n_batches,
            "acc": correct / total if total > 0 else 0.0,
        }

    def fit(self):
        print("Starting Classifier Training...")
        best_val_acc = 0.0

        epochs = Config.CLASSIFIER_EPOCHS
        if Config.DEBUG:
            epochs = 2

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate()

            duration = time.time() - start_time

            print(f"Epoch {epoch}/{epochs} | Time: {duration:.2f}s")
            print(
                f"  Train Loss: {train_metrics['loss']:.6f} | Acc: {train_metrics['acc']:.6f}"
            )
            print(
                f"  Val Loss:   {val_metrics['loss']:.6f} | Acc: {val_metrics['acc']:.6f}"
            )

            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                torch.save(self.model.state_dict(), Config.CLASSIFIER_CHECKPOINT)
                print(f"  -> Saved Best Classifier Model (Acc: {best_val_acc:.6f})")

        print("Classifier Training Completed.")

import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    seed_everything,
    collate_fn,
    AverageMeter,
    MeanAveragePrecision,
)
from library.dataset import CovidDataset
from library.model import MultiTaskEfficientDet


class Trainer:
    """
    Trainer class for the Multi-Task EfficientDet model.
    Handles training, validation, early stopping, and submission generation.
    """

    def __init__(self, config=Config):
        self.config = config
        self.device = torch.device(config.DEVICE)

        # Ensure reproducibility
        seed_everything(config.SEED)

        # Initialize Model
        self.model = MultiTaskEfficientDet(config).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.T_MAX, eta_min=config.ETA_MIN
        )

        # Training State
        self.best_map = 0.0
        self.patience_counter = 0

    def train_one_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loss_meter = AverageMeter()

        for images, targets, _ in train_loader:
            images = images.to(self.device)
            # Move targets to device (handling non-tensor data like strings)
            targets = [
                {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()
                }
                for t in targets
            ]

            # Forward pass (returns dict of losses in training mode)
            loss_dict = self.model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_meter.update(loss.item(), images.size(0))

        return loss_meter.avg

    def validate(self, val_loader):
        """
        Runs validation and calculates mAP.
        """
        self.model.eval()
        map_metric = MeanAveragePrecision(num_classes=self.config.NUM_DETECTION_CLASSES)

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(self.device)
                targets = [
                    {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()
                    }
                    for t in targets
                ]

                # Forward pass (returns detections in eval mode)
                detections = self.model(images)

                # Extract data for mAP calculation
                pred_boxes = [d["boxes"] for d in detections]
                pred_scores = [d["scores"] for d in detections]
                pred_labels = [d["labels"] for d in detections]

                gt_boxes = [t["boxes"] for t in targets]
                gt_labels = [t["labels"] for t in targets]

                map_metric.update(
                    pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels
                )

        return map_metric.compute()

    def fit(self, epochs=None, debug=False):
        """
        Main training loop with Early Stopping.

        Args:
            epochs (int, optional): Number of epochs to train. Defaults to Config.EPOCHS.
            debug (bool): If True, uses a small subset of data for quick debugging.
        """
        if epochs is None:
            epochs = self.config.EPOCHS

        # Load Datasets
        train_dataset = CovidDataset("train", load_cached_data=True)
        val_dataset = CovidDataset("val", load_cached_data=True)

        if debug:
            print("Debug mode: Using subset of data.")
            train_dataset.data = train_dataset.data[:100]
            val_dataset.data = val_dataset.data[:50]

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_map = self.validate(val_loader)

            self.scheduler.step()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val mAP: {val_map}"
            )

            # Early Stopping and Checkpointing
            if val_map > self.best_map:
                self.best_map = val_map
                self.patience_counter = 0
                torch.save(self.model.state_dict(), self.config.BEST_MODEL_PATH)
                print(f"New best model saved with mAP: {self.best_map}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.config.PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    def predict(self):
        """
        Generates predictions for the test set and saves the submission file.
        """
        if not os.path.exists(self.config.BEST_MODEL_PATH):
            print(f"Error: Best model not found at {self.config.BEST_MODEL_PATH}")
            return

        print("Loading best model for inference...")
        self.model.load_state_dict(
            torch.load(self.config.BEST_MODEL_PATH, map_location=self.device)
        )
        self.model.eval()

        test_dataset = CovidDataset("test", load_cached_data=True)
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            collate_fn=collate_fn,
        )

        study_preds = {}  # study_id -> list of probability arrays
        image_preds = {}  # image_id -> prediction string

        print("Generating predictions...")
        with torch.no_grad():
            for images, targets, ids in test_loader:
                images = images.to(self.device)

                detections = self.model(images)

                for i, det in enumerate(detections):
                    img_id = ids[i]
                    study_id = targets[i]["study_id"]

                    # 1. Collect Study Predictions
                    probs = det["study_probs"].cpu().numpy()
                    if study_id not in study_preds:
                        study_preds[study_id] = []
                    study_preds[study_id].append(probs)

                    # 2. Generate Image Predictions
                    boxes = det["boxes"].cpu().numpy()
                    scores = det["scores"].cpu().numpy()

                    # Determine if we should predict "none" based on the "Negative for Pneumonia" probability
                    # Index 0 corresponds to "Negative for Pneumonia"
                    neg_prob = probs[0]

                    if neg_prob > self.config.STUDY_CONF_THRESHOLD:
                        pred_str = "none 1 0 0 1 1"
                    else:
                        s = []
                        for b, sc in zip(boxes, scores):
                            # Format: opacity confidence xmin ymin xmax ymax
                            s.append(
                                f"opacity {sc:.4f} {b[0]:.1f} {b[1]:.1f} {b[2]:.1f} {b[3]:.1f}"
                            )

                        if not s:
                            pred_str = "none 1 0 0 1 1"
                        else:
                            pred_str = " ".join(s)

                    image_preds[img_id] = pred_str

        # 3. Format Study Level Predictions
        study_rows = []
        # Mapping indices to submission strings
        # 0: Negative, 1: Typical, 2: Indeterminate, 3: Atypical
        idx_to_label = {0: "negative", 1: "typical", 2: "indeterminate", 3: "atypical"}

        for study_id, probs_list in study_preds.items():
            # Average probabilities across all images in the study
            avg_probs = np.mean(probs_list, axis=0)

            # Select the class with the highest probability
            idx = np.argmax(avg_probs)
            label = idx_to_label[idx]
            conf = avg_probs[idx]

            # Format: class_id confidence 0 0 1 1
            pred_string = f"{label} {conf:.4f} 0 0 1 1"
            study_rows.append(
                {"id": f"{study_id}_study", "PredictionString": pred_string}
            )

        # 4. Format Image Level Predictions
        image_rows = [
            {"id": f"{k}_image", "PredictionString": v} for k, v in image_preds.items()
        ]

        # 5. Combine and Save
        df_study = pd.DataFrame(study_rows)
        df_image = pd.DataFrame(image_rows)
        df_sub = pd.concat([df_study, df_image], ignore_index=True)

        # Ensure submission directory exists
        os.makedirs(os.path.dirname(self.config.SUBMISSION_PATH), exist_ok=True)

        df_sub.to_csv(self.config.SUBMISSION_PATH, index=False)
        print(f"Submission saved to {self.config.SUBMISSION_PATH}")

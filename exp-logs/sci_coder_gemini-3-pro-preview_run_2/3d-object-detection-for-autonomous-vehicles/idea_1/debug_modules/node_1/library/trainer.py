import os
import torch
import torch.optim as optim
import numpy as np
from library.config import Config, set_random_seed
from library.model import PointPillars
from library.loss import LossModule
from library.data_loader import create_data_loaders


class Trainer:
    """
    Manages the training and validation lifecycle of the 3D Object Detection model.
    """

    def __init__(self, config=None, load_cached_data=True):
        self.config = config if config is not None else Config
        set_random_seed(self.config.SEED)

        # Setup directories
        self.working_dir = os.path.join(self.config.WORKING_DIR, "idea_1")
        os.makedirs(self.working_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.working_dir, "best_model.pth")

        # Device configuration
        self.device = torch.device(self.config.DEVICE)

        # Initialize Data Loaders
        # This handles caching via the data_loader library
        self.loaders = create_data_loaders(
            self.config, load_cached_data=load_cached_data
        )

        # Initialize Model
        self.model = PointPillars(self.config).to(self.device)

        # Initialize Loss
        self.criterion = LossModule(self.config).to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Initialize Scheduler
        train_loader = self.loaders.get("train")
        if train_loader:
            self.scheduler = optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.LEARNING_RATE,
                epochs=self.config.EPOCHS,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,
                div_factor=10,
                final_div_factor=100,
            )
        else:
            self.scheduler = None

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        loader = self.loaders.get("train")
        if not loader:
            return {}

        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0

        for batch in loader:
            # Move inputs to device
            # Points is a list of tensors (variable size point clouds)
            points = [p.to(self.device) for p in batch["points"]]

            # Move targets to device
            if "boxes" not in batch or "labels" not in batch:
                continue

            gt_boxes = [b.to(self.device) for b in batch["boxes"]]
            gt_labels = [l.to(self.device) for l in batch["labels"]]

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            cls_preds, reg_preds = self.model(points)

            if cls_preds is None:
                continue

            # Compute loss
            loss, loss_dict = self.criterion(cls_preds, reg_preds, gt_boxes, gt_labels)

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.GRAD_CLIP_NORM
            )

            # Optimizer step
            self.optimizer.step()

            # Scheduler step
            if self.scheduler:
                self.scheduler.step()

            # Accumulate metrics
            total_loss += loss.item()
            cls_loss_sum += loss_dict["cls_loss"]
            reg_loss_sum += loss_dict["reg_loss"]
            num_batches += 1

        if num_batches == 0:
            return {}

        return {
            "loss": total_loss / num_batches,
            "cls_loss": cls_loss_sum / num_batches,
            "reg_loss": reg_loss_sum / num_batches,
        }

    def validate(self):
        """
        Runs validation on the validation set.
        """
        self.model.eval()
        loader = self.loaders.get("val")
        if not loader:
            return {}

        total_loss = 0.0
        cls_loss_sum = 0.0
        reg_loss_sum = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in loader:
                points = [p.to(self.device) for p in batch["points"]]

                # Validation targets
                if "boxes" not in batch or "labels" not in batch:
                    continue

                gt_boxes = [b.to(self.device) for b in batch["boxes"]]
                gt_labels = [l.to(self.device) for l in batch["labels"]]

                # Forward pass
                cls_preds, reg_preds = self.model(points)

                if cls_preds is None:
                    continue

                # Compute loss
                loss, loss_dict = self.criterion(
                    cls_preds, reg_preds, gt_boxes, gt_labels
                )

                total_loss += loss.item()
                cls_loss_sum += loss_dict["cls_loss"]
                reg_loss_sum += loss_dict["reg_loss"]
                num_batches += 1

        if num_batches == 0:
            return {}

        return {
            "loss": total_loss / num_batches,
            "cls_loss": cls_loss_sum / num_batches,
            "reg_loss": reg_loss_sum / num_batches,
        }

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        print("Starting training process...")
        best_val_loss = float("inf")
        patience = 3
        patience_counter = 0

        for epoch in range(1, self.config.EPOCHS + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()

            # Print full precision metrics
            print(f"Epoch {epoch}")
            print(f"Train Metrics: {train_metrics}")
            print(f"Val Metrics: {val_metrics}")

            # Early Stopping Check
            val_loss = val_metrics.get("loss", float("inf"))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                print(
                    f"Validation loss improved. Saved model to {self.checkpoint_path}"
                )
            else:
                patience_counter += 1
                print(
                    f"Validation loss did not improve. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print("Training complete.")

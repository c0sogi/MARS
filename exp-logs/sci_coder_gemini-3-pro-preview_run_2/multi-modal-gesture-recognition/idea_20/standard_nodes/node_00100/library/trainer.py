import time
import torch
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import (
    get_device,
    save_checkpoint,
    AverageMeter,
    set_seed,
    log_metrics,
    count_parameters,
)
from library.data_loader import get_dataloaders
from library.model import DSG_CRCN
from library.loss import DSG_Loss


class Trainer:
    """
    Trainer class for the Dual-Scale Supervised Gated-Cascaded Recurrent-Convolutional Network.
    """

    def __init__(self):
        """
        Initialize the Trainer with model, optimizer, loss, and data loaders.
        """
        self.device = get_device()

        # Initialize Model
        self.model = DSG_CRCN().to(self.device)
        print(
            f"Model initialized with {count_parameters(self.model)} trainable parameters."
        )

        # Initialize Loss Function
        self.criterion = DSG_Loss().to(self.device)

        # Initialize Optimizer
        # We separate parameters into those with weight decay (weights) and without (biases/norms)
        # though AdamW usually handles this, explicit separation is good practice.
        # Here we stick to simple AdamW as per prompt specs.
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Data Loaders
        self.train_loader, self.val_loader, _ = get_dataloaders(
            batch_size=Config.BATCH_SIZE
        )

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()

        # Meters for tracking metrics
        loss_meter = AverageMeter("Loss")
        cls_acc_meter = AverageMeter("ClsAcc")

        # Stage-specific loss meters
        stage1_loss_meter = AverageMeter("S1_Loss")
        stage2_loss_meter = AverageMeter("S2_Loss")
        stage3_loss_meter = AverageMeter("S3_Loss")

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            mask = batch["mask"].to(self.device)

            targets = {"labels": labels, "boundaries": boundaries, "mask": mask}

            # Forward Pass
            outputs = self.model(features, mask)

            # Compute Loss
            loss, metrics = self.criterion(outputs, targets)

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient Clipping (Optional but recommended for RNNs)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Update Meters
            loss_meter.update(loss.item(), features.size(0))
            stage1_loss_meter.update(metrics["stage1_loss"], features.size(0))
            stage2_loss_meter.update(metrics["stage2_loss"], features.size(0))
            stage3_loss_meter.update(metrics["stage3_loss"], features.size(0))

            # Calculate Training Accuracy (Stage 3)
            # outputs['stage3_cls'] is (B, T, C) probabilities
            pred_probs = outputs["stage3_cls"]
            pred_labels = torch.argmax(pred_probs, dim=2)  # (B, T)

            # Mask out padding for accuracy calculation
            correct = (pred_labels == labels) * mask
            acc = correct.sum() / (mask.sum() + 1e-7)
            cls_acc_meter.update(acc.item(), features.size(0))

        return {
            "loss": loss_meter.avg,
            "acc": cls_acc_meter.avg,
            "s1_loss": stage1_loss_meter.avg,
            "s2_loss": stage2_loss_meter.avg,
            "s3_loss": stage3_loss_meter.avg,
        }

    def validate(self, epoch):
        """
        Runs validation on the validation set.
        """
        self.model.eval()

        loss_meter = AverageMeter("ValLoss")
        cls_acc_meter = AverageMeter("ValAcc")

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)
                mask = batch["mask"].to(self.device)

                targets = {"labels": labels, "boundaries": boundaries, "mask": mask}

                outputs = self.model(features, mask)
                loss, _ = self.criterion(outputs, targets)

                loss_meter.update(loss.item(), features.size(0))

                # Calculate Validation Accuracy (Stage 3)
                pred_probs = outputs["stage3_cls"]
                pred_labels = torch.argmax(pred_probs, dim=2)

                correct = (pred_labels == labels) * mask
                acc = correct.sum() / (mask.sum() + 1e-7)
                cls_acc_meter.update(acc.item(), features.size(0))

        return {"val_loss": loss_meter.avg, "val_acc": cls_acc_meter.avg}

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        set_seed(Config.SEED)

        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.NUM_EPOCHS} epochs...")

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            start_time = time.time()

            # Train
            train_metrics = self.train_epoch(epoch)

            # Validate
            val_metrics = self.validate(epoch)

            epoch_time = time.time() - start_time

            # Combine metrics for logging
            all_metrics = {**train_metrics, **val_metrics, "time": epoch_time}
            log_metrics(epoch, all_metrics)

            # Checkpoint & Early Stopping
            current_val_loss = val_metrics["val_loss"]
            is_best = current_val_loss < best_val_loss

            if is_best:
                best_val_loss = current_val_loss
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "state_dict": self.model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                        "best_metric": best_val_loss,
                    },
                    is_best=True,
                    checkpoint_dir=Config.CACHE_DIR,
                )
                print(f"New best model saved with val_loss: {best_val_loss}")
            else:
                patience_counter += 1
                print(
                    f"No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print("Training complete.")

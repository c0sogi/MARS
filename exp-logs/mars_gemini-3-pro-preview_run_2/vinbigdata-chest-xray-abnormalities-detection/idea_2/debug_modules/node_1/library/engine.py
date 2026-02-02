import torch
import math
import sys
import time
import numpy as np
import pandas as pd
import os
import torchvision
from library.config import Config
from library.utils import get_logger

logger = get_logger("engine")


class Trainer:
    """
    Encapsulates training, validation, and inference logic for Faster R-CNN.
    """

    def __init__(self, model, optimizer, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.best_val_loss = float("inf")
        self.patience_counter = 0

    def train_one_epoch(self, data_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(data_loader)

        logger.info(
            f"Epoch {epoch} - Starting Training Loop over {num_batches} batches..."
        )

        for batch_idx, (images, targets) in enumerate(data_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Forward pass
            # In train mode, Faster R-CNN returns a dict of losses
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                logger.error(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            total_loss += loss_value

            # Optional: Step scheduler per iteration if needed, but we usually do per epoch

        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} - Training Loss: {avg_loss}")
        return avg_loss

    @torch.no_grad()
    def evaluate(self, data_loader):
        """
        Evaluates the model on the validation set.
        Uses 'train' mode with no_grad to compute validation loss.
        """
        # To get loss, we must be in train mode.
        # In eval mode, the model returns predictions, not losses.
        self.model.train()
        total_loss = 0.0
        num_batches = len(data_loader)

        logger.info(f"Starting Validation Loop over {num_batches} batches...")

        for images, targets in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

        avg_loss = total_loss / num_batches
        logger.info(f"Validation Loss: {avg_loss}")
        return avg_loss

    def fit(self, train_loader, val_loader, epochs, save_path):
        """
        Runs the full training pipeline with Early Stopping.
        """
        logger.info(f"Starting training for {epochs} epochs on device {self.device}")

        for epoch in range(1, epochs + 1):
            # Train
            train_loss = self.train_one_epoch(train_loader, epoch)

            # Validate
            val_loss = self.evaluate(val_loader)

            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()

            # Early Stopping and Checkpointing
            if val_loss < self.best_val_loss:
                logger.info(
                    f"Validation loss improved from {self.best_val_loss} to {val_loss}. Saving model..."
                )
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), save_path)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                logger.info(
                    f"Validation loss did not improve. Patience: {self.patience_counter}/{Config.PATIENCE}"
                )
                if self.patience_counter >= Config.PATIENCE:
                    logger.info("Early stopping triggered.")
                    break

        logger.info(f"Training completed. Best Validation Loss: {self.best_val_loss}")

    @torch.no_grad()
    def predict(self, data_loader, output_path):
        """
        Runs inference on the test set and saves predictions to CSV.
        """
        self.model.eval()
        results = []

        logger.info("Starting Inference...")

        # Ensure CPU device for post-processing
        cpu_device = torch.device("cpu")

        for images, targets in data_loader:
            # Move images to device
            images = list(img.to(self.device) for img in images)

            # Forward pass
            outputs = self.model(images)

            # Move outputs to CPU
            outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

            # Process batch
            for i, output in enumerate(outputs):
                image_id = targets[i]["img_id_str"]

                # Get scaling factors to restore original bbox coordinates
                scale_x = targets[i]["scale_x"].item()
                scale_y = targets[i]["scale_y"].item()

                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # 1. Filter by confidence threshold
                keep_idxs = scores > Config.CONFIDENCE_THRESHOLD
                boxes = boxes[keep_idxs]
                scores = scores[keep_idxs]
                labels = labels[keep_idxs]

                # 2. Apply NMS (Non-Maximum Suppression)
                # torchvision.ops.nms performs NMS class-agnostically or we can loop per class.
                # Here we use batched_nms to handle multi-class NMS efficiently.
                if len(boxes) > 0:
                    keep_nms = torchvision.ops.batched_nms(
                        boxes, scores, labels, Config.IOU_THRESHOLD
                    )
                    boxes = boxes[keep_nms]
                    scores = scores[keep_nms]
                    labels = labels[keep_nms]

                prediction_strings = []

                if len(boxes) > 0:
                    for box, score, label in zip(boxes, scores, labels):
                        # Rescale box to original image dimensions
                        xmin = box[0].item() / scale_x
                        ymin = box[1].item() / scale_y
                        xmax = box[2].item() / scale_x
                        ymax = box[3].item() / scale_y

                        # Map Model Label back to Class ID
                        # Model Label 1..14 -> Class ID 0..13
                        # Model Label 0 is background (not output here)
                        class_id = int(label.item()) - 1

                        conf = score.item()

                        prediction_strings.append(
                            f"{class_id} {conf:.4f} {xmin:.4f} {ymin:.4f} {xmax:.4f} {ymax:.4f}"
                        )

                    prediction_string = " ".join(prediction_strings)
                else:
                    # No finding
                    prediction_string = "14 1 0 0 1 1"

                results.append(
                    {"image_id": image_id, "PredictionString": prediction_string}
                )

        # Create DataFrame
        submission_df = pd.DataFrame(results)

        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        submission_df.to_csv(output_path, index=False)
        logger.info(f"Submission saved to {output_path}")


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    """
    Wrapper function to maintain compatibility if needed,
    but logic is encapsulated in Trainer class.
    """
    trainer = Trainer(model, optimizer, device)
    return trainer.train_one_epoch(data_loader, epoch)


def evaluate(model, data_loader, device):
    """
    Wrapper function for evaluation.
    """
    trainer = Trainer(model, None, device)
    return trainer.evaluate(data_loader)

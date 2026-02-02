import torch
import math
import sys
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from library.config import Config, CLASS_ID_TO_LABEL
from library.utils import format_prediction_string


class Engine:
    def __init__(self, model, device, optimizer=None, scheduler=None):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train_one_epoch(self, data_loader, epoch):
        self.model.train()
        final_loss = 0.0
        count = 0

        # Iterate over data
        # Note: We avoid tqdm here to keep logs clean as per instructions,
        # but simple print statements for epoch progress are allowed.
        for images, targets, image_ids in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()

            final_loss += loss_value
            count += 1

        if self.scheduler:
            self.scheduler.step()

        avg_loss = final_loss / count if count > 0 else 0
        return avg_loss

    @torch.no_grad()
    def evaluate_loss(self, data_loader):
        # To get loss from torchvision detection models, we must be in train mode
        # even during validation, but with no_grad.
        self.model.train()
        total_loss = 0.0
        count = 0

        for images, targets, image_ids in data_loader:
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()
            count += 1

        return total_loss / count if count > 0 else 0

    def fit_model(self, train_loader, val_loader, epochs, patience=3):
        best_loss = float("inf")
        patience_counter = 0

        # Ensure working directory exists
        os.makedirs(Config.WORKING_DIR, exist_ok=True)
        save_path = os.path.join(Config.WORKING_DIR, "best_model.pth")

        print(f"Starting training for {epochs} epochs on {self.device}...")

        for epoch in range(epochs):
            train_loss = self.train_one_epoch(train_loader, epoch)
            val_loss = self.evaluate_loss(val_loader)

            print(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
            )

            # Early Stopping and Checkpointing
            if val_loss < best_loss:
                best_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), save_path)
                print(f"Validation loss improved. Model saved to {save_path}")
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("Early stopping triggered.")
                    break

        print("Training complete.")
        return save_path

    @torch.no_grad()
    def inference(self, test_loader, model_path):
        print("Starting inference...")

        # Load best model
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded model from {model_path}")
        else:
            print(
                f"Warning: Model path {model_path} does not exist. Using current model weights."
            )

        self.model.eval()
        self.model.to(self.device)

        results = []

        # Threshold for keeping a box
        detection_threshold = Config.DETECTION_THRESHOLD

        for images, image_ids in test_loader:
            images = list(img.to(self.device) for img in images)

            # Forward pass returns list of dicts: [{'boxes':..., 'labels':..., 'scores':...}, ...]
            outputs = self.model(images)

            # Move to CPU for processing
            outputs = [{k: v.cpu().numpy() for k, v in t.items()} for t in outputs]

            for i, output in enumerate(outputs):
                image_id = image_ids[i]

                boxes = output["boxes"]
                scores = output["scores"]
                labels = output["labels"]

                # Filter by threshold
                mask = scores >= detection_threshold
                boxes = boxes[mask]
                scores = scores[mask]
                labels = labels[mask]

                # --- 1. Study Level Prediction ---
                # Logic:
                # If no boxes -> "negative 1 0 0 1 1"
                # Else -> find max score box, map its class to study label.
                # Format: "{label} {score} 0 0 1 1"

                study_id = f"{image_id}_study"

                if len(boxes) == 0:
                    study_pred = "negative 1 0 0 1 1"
                else:
                    # Find box with max score
                    max_idx = np.argmax(scores)
                    max_score = scores[max_idx]
                    max_label_id = labels[max_idx]

                    # Map ID to string (1: typical, 2: indeterminate, 3: atypical)
                    # Note: If model predicts 0 (background) or something else, fallback to negative
                    label_str = CLASS_ID_TO_LABEL.get(max_label_id, "negative")

                    if label_str == "negative":
                        study_pred = "negative 1 0 0 1 1"
                    else:
                        study_pred = f"{label_str} {max_score:.6f} 0 0 1 1"

                results.append({"id": study_id, "PredictionString": study_pred})

                # --- 2. Image Level Prediction ---
                # Logic:
                # If no boxes -> "none 1 0 0 1 1"
                # Else -> All boxes are "opacity".
                # Format: "opacity {score} {xmin} {ymin} {xmax} {ymax} ..."

                img_pred_id = f"{image_id}_image"

                if len(boxes) == 0:
                    image_pred = "none 1 0 0 1 1"
                else:
                    # Create lists for formatter
                    # Class is always "opacity" for image level
                    pred_labels = ["opacity"] * len(boxes)
                    pred_scores = scores.tolist()
                    pred_boxes = boxes.tolist()  # [xmin, ymin, xmax, ymax]

                    image_pred = format_prediction_string(
                        pred_labels, pred_boxes, pred_scores
                    )

                results.append({"id": img_pred_id, "PredictionString": image_pred})

        # Save Submission
        submission_df = pd.DataFrame(results)

        # Ensure submission directory exists
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)

        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        return submission_df

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import pandas as pd
from library.utils import set_seed, calculate_iou_batch, rle_encode

# Set fixed random seed for reproducibility
set_seed(42)


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation.
    Calculates 1 - Dice Coefficient.
    """

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Apply sigmoid to logits to get probabilities
        probs = torch.sigmoid(logits)

        # Flatten the tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice


class SaltTrainer:
    """
    Trainer class encapsulating training, validation, and inference logic for Salt Segmentation.
    """

    def __init__(
        self, model, device, learning_rate=1e-3, checkpoint_dir="./working/idea_1"
    ):
        """
        Args:
            model (nn.Module): The neural network model.
            device (torch.device): Device to run training on (CPU/GPU).
            learning_rate (float): Learning rate for the optimizer.
            checkpoint_dir (str): Directory to save model checkpoints.
        """
        self.model = model
        self.device = device
        self.checkpoint_dir = checkpoint_dir

        # Ensure checkpoint directory exists
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Initialize Scheduler
        # Cite {solution_lesson_node_00006}
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5
        )

        # Initialize Loss Functions
        self.bce_criterion = nn.BCEWithLogitsLoss()
        self.dice_criterion = DiceLoss()

        # Track best performance
        self.best_val_iou = -float("inf")
        self.best_model_path = os.path.join(self.checkpoint_dir, "best_model.pth")

    def criterion(self, logits, targets):
        """
        Composite loss function: BCE + Dice.
        """
        bce = self.bce_criterion(logits, targets)
        dice = self.dice_criterion(logits, targets)
        return bce + dice

    def train_epoch(self, train_loader):
        """
        Runs training for one epoch.
        """
        self.model.train()
        running_loss = 0.0

        for inputs, masks, _ in train_loader:
            inputs = inputs.to(self.device)
            masks = masks.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(inputs)
            loss = self.criterion(logits, masks)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        return epoch_loss

    def validate(self, val_loader):
        """
        Runs evaluation on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        total_iou = 0.0
        total_count = 0

        with torch.no_grad():
            for inputs, masks, _ in val_loader:
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                batch_size = inputs.size(0)

                # Standard forward pass for loss calculation
                logits = self.model(inputs)
                loss = self.criterion(logits, masks)
                running_loss += loss.item() * batch_size

                # Test Time Augmentation (TTA) for IoU calculation
                # 1. Original
                probs_orig = torch.sigmoid(logits)

                # 2. Horizontal Flip
                inputs_flip = torch.flip(inputs, [3])
                logits_flip = self.model(inputs_flip)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip_back = torch.flip(probs_flip, [3])

                # Average
                probs = (probs_orig + probs_flip_back) / 2.0

                # Calculate IoU on TTA probabilities
                batch_iou = calculate_iou_batch(probs, masks, threshold=0.5)

                # Accumulate weighted IoU
                total_iou += batch_iou * batch_size
                total_count += batch_size

        val_loss = running_loss / len(val_loader.dataset)
        val_iou = total_iou / total_count if total_count > 0 else 0.0

        return val_loss, val_iou

    def train(self, train_loader, val_loader, epochs=50, patience=10):
        """
        Main training loop with Early Stopping and Checkpointing.
        """
        print(f"Starting training for {epochs} epochs with patience {patience}...")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_iou = self.validate(val_loader)

            # Step the scheduler based on Validation Loss
            # Cite {solution_lesson_node_00006}
            self.scheduler.step(val_loss)

            # Print metrics with full precision
            print(
                f"Epoch {epoch}/{epochs} - Train Loss: {train_loss} - Val Loss: {val_loss} - Val IoU: {val_iou}"
            )

            # Checkpointing and Early Stopping Logic
            if val_iou > self.best_val_iou:
                self.best_val_iou = val_iou
                patience_counter = 0
                torch.save(self.model.state_dict(), self.best_model_path)
                print(f"New best model saved with IoU: {val_iou}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training finished. Best Val IoU: {self.best_val_iou}")

    def load_best_model(self):
        """
        Loads the weights of the best performing model.
        """
        if os.path.exists(self.best_model_path):
            self.model.load_state_dict(
                torch.load(self.best_model_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.best_model_path}")
        else:
            print(f"Warning: No checkpoint found at {self.best_model_path}")

    def generate_submission(
        self, test_loader, output_file="./submission/submission.csv"
    ):
        """
        Generates predictions for the test set and saves them to a CSV file.
        Handles cropping from 128x128 back to 101x101.
        """
        print("Generating submission...")
        self.load_best_model()
        self.model.eval()

        predictions = []
        ids = []

        # Padding parameters derived from SaltDataset logic
        # Original: 101x101, Target: 128x128
        # Pad Total: 27. Top: 13, Bottom: 14, Left: 13, Right: 14.
        crop_top = 13
        crop_left = 13
        orig_h = 101
        orig_w = 101

        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with torch.no_grad():
            for inputs, image_ids in test_loader:
                inputs = inputs.to(self.device)

                # Test Time Augmentation (TTA)
                # 1. Original
                logits = self.model(inputs)
                probs_orig = torch.sigmoid(logits)

                # 2. Horizontal Flip
                inputs_flip = torch.flip(inputs, [3])
                logits_flip = self.model(inputs_flip)
                probs_flip = torch.sigmoid(logits_flip)
                probs_flip_back = torch.flip(probs_flip, [3])

                # Average
                probs = (probs_orig + probs_flip_back) / 2.0

                # Convert to numpy for post-processing
                probs_np = probs.detach().cpu().numpy()  # Shape: (Batch, 1, 128, 128)

                for i in range(len(image_ids)):
                    img_id = image_ids[i]
                    prob_map = probs_np[i, 0]  # Shape: (128, 128)

                    # Crop center to original size
                    prob_map_cropped = prob_map[
                        crop_top : crop_top + orig_h, crop_left : crop_left + orig_w
                    ]

                    # Binarize with threshold 0.5
                    mask = (prob_map_cropped > 0.5).astype(np.uint8)

                    # Encode to RLE
                    rle = rle_encode(mask)

                    ids.append(img_id)
                    predictions.append(rle)

        # Create DataFrame and save to CSV
        df = pd.DataFrame({"id": ids, "rle_mask": predictions})
        df.to_csv(output_file, index=False)
        print(f"Submission saved to {output_file}")

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import RCMCN
from library.dataset import GestureDataset
from library.utils import (
    decode_predictions_to_labels,
    calculate_score,
    run_length_encoding,
)


class SmoothingLoss(nn.Module):
    """
    Log-Space Smoothing Loss: Truncated MSE on adjacent log-probabilities.
    Enforces temporal smoothness in predictions.
    """

    def __init__(self, threshold=1.0):
        super(SmoothingLoss, self).__init__()
        self.threshold = threshold

    def forward(self, logits):
        """
        Args:
            logits: (Batch, Time, Classes)
        Returns:
            Scalar loss
        """
        # Convert to log-probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)

        # Calculate difference between adjacent frames: P[t] - P[t-1]
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Truncated Squared Error: min(diff^2, threshold^2)
        loss = torch.clamp(diff**2, max=self.threshold**2)

        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Cascaded Loss Function for Deep Supervision.
    L_total = L_CE(P1) + L_CE(P2) + L_CE(P3) + lambda * (L_smooth(P2) + L_smooth(P3))
    """

    def __init__(self, device):
        super(CombinedLoss, self).__init__()

        # Weighted Cross Entropy
        # Background class (0) gets lower weight
        class_weights = torch.ones(Config.NUM_CLASSES)
        class_weights[0] = Config.BACKGROUND_CLASS_WEIGHT
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

        # Smoothing Loss
        self.smooth_criterion = SmoothingLoss(threshold=Config.SMOOTHING_THRESHOLD)
        self.smoothing_weight = Config.SMOOTHING_WEIGHT

    def forward(self, outputs, targets):
        """
        Args:
            outputs: Tuple (logits1, logits2, logits3) from RCMCN
            targets: (Batch, Time) LongTensor of labels
        """
        logits1, logits2, logits3 = outputs

        # Flatten for CrossEntropy: (Batch * Time, Classes) vs (Batch * Time)
        # Reshape targets to match logits
        targets_flat = targets.view(-1)

        ce_loss1 = self.ce_criterion(
            logits1.reshape(-1, Config.NUM_CLASSES), targets_flat
        )
        ce_loss2 = self.ce_criterion(
            logits2.reshape(-1, Config.NUM_CLASSES), targets_flat
        )
        ce_loss3 = self.ce_criterion(
            logits3.reshape(-1, Config.NUM_CLASSES), targets_flat
        )

        # Smoothing Loss (Applied to refined stages only)
        smooth_loss2 = self.smooth_criterion(logits2)
        smooth_loss3 = self.smooth_criterion(logits3)

        # Total Loss
        total_loss = (ce_loss1 + ce_loss2 + ce_loss3) + self.smoothing_weight * (
            smooth_loss2 + smooth_loss3
        )

        return total_loss


class Trainer:
    """
    Handles training, validation, and checkpointing for RC-MCN.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader  # Note: batch_size=1 for variable length sequences
        self.device = device

        self.criterion = CombinedLoss(device)
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.best_score = float("inf")
        self.patience_counter = 0
        self.patience_limit = 10  # Early stopping patience

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns tuple of 3 logits)
            outputs = self.model(features)

            # Compute loss
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        """
        Validates the model on the validation set using Levenshtein distance.
        Since validation samples vary in length, we process them one by one (Batch Size = 1).
        """
        self.model.eval()

        ground_truth_dict = {}
        predictions_dict = {}

        with torch.no_grad():
            # Iterate over validation dataset
            # We assume val_loader has batch_size=1 and shuffle=False
            for i, (features, labels) in enumerate(self.val_loader):
                features = features.to(self.device)
                # labels: (1, T)

                # Get sample ID from dataset directly (assuming order is preserved)
                # The loader returns tensors, so we look up metadata in the dataset object
                sample_id = self.val_loader.dataset.data[i]["sample_id"]

                # Forward pass
                _, _, logits3 = self.model(features)

                # Get probabilities for final stage
                probs = torch.softmax(logits3, dim=2)

                # Decode predictions: (1, T, C) -> (T, C) -> List of IDs
                frame_probs = probs.squeeze(0).cpu().numpy()
                pred_labels = decode_predictions_to_labels(frame_probs)
                predictions_dict[sample_id] = pred_labels

                # Decode Ground Truth from frame-wise labels
                # We use the same RLE logic to extract segments from the dense label array
                gt_frame_labels = labels.squeeze(0).cpu().numpy()
                gt_segments = run_length_encoding(gt_frame_labels)
                gt_labels = [seg["label"] for seg in gt_segments if seg["label"] != 0]
                ground_truth_dict[sample_id] = gt_labels

        # Calculate metric
        score = calculate_score(ground_truth_dict, predictions_dict)
        return score

    def fit(self, epochs):
        print(f"Starting training for {epochs} epochs...")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            val_score = self.validate()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (Levenshtein Error): {val_score:.10f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpointing & Early Stopping
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.BEST_MODEL_PATH)
                # print(f"  New best model saved! Score: {self.best_score:.10f}")
            else:
                self.patience_counter += 1

            if self.patience_counter >= self.patience_limit:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Validation Score: {self.best_score:.10f}")


def train_model(load_cached_data=True):
    """
    Main entry point to train the RC-MCN model.
    """
    # 1. Setup
    Config.set_seed(Config.SEED)
    Config.setup_dirs()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Data Loading
    # Train: Sliding windows, shuffled, batched
    train_dataset = GestureDataset(
        split="train", load_cached_data=load_cached_data, transform=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    # Val: Full sequences, no shuffle, batch_size=1
    val_dataset = GestureDataset(
        split="val", load_cached_data=load_cached_data, transform=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True
    )

    print(f"Train samples (windows): {len(train_dataset)}")
    print(f"Val samples (sequences): {len(val_dataset)}")

    # 3. Model Initialization
    model = RCMCN().to(device)

    # 4. Training
    trainer = Trainer(model, train_loader, val_loader, device)
    trainer.fit(Config.EPOCHS)

    return trainer.best_score

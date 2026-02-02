import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from library.config import Config
from library.model import PAKRNet
from library.data_loader import get_dataloaders
from library.utils import process_predictions, compute_levenshtein


class CombinedLoss(nn.Module):
    """
    Cascaded Loss Function for PAK-RN.
    Combines:
    1. Weighted Cross-Entropy (handling class imbalance).
    2. Log-Space Smoothing Loss (temporal consistency).
    3. Deep Supervision (summing losses from all 3 stages).
    """

    def __init__(self, device):
        super(CombinedLoss, self).__init__()

        # Weighted Cross-Entropy
        # Background (class 0) gets weight 0.2, others 1.0
        weights = torch.ones(Config.NUM_CLASSES, device=device)
        weights[0] = Config.LOSS_BG_WEIGHT
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)

        self.smooth_lambda = Config.SMOOTHING_LAMBDA
        self.smooth_threshold = Config.SMOOTHING_THRESHOLD

    def log_space_smoothing_loss(self, log_probs):
        """
        Computes Truncated MSE on adjacent log-probabilities.
        Args:
            log_probs: (Batch, Time, Classes)
        """
        # Calculate difference between adjacent frames: t and t-1
        diff = log_probs[:, 1:, :] - log_probs[:, :-1, :]

        # Truncate (Clamp) the gradients/differences
        diff = torch.clamp(diff, min=-self.smooth_threshold, max=self.smooth_threshold)

        # MSE
        loss = torch.mean(diff**2)
        return loss

    def forward(self, outputs, targets):
        """
        Args:
            outputs: List of [logits_1, logits_2, logits_3] from the model.
                     Each shape: (Batch, Time, Classes)
            targets: (Batch, Time) LongTensor
        """
        # Transpose logits for CrossEntropy: (B, T, C) -> (B, C, T)
        logits_1, logits_2, logits_3 = outputs

        # Stage 1 Loss: Pure Cross Entropy
        loss_1 = self.ce_loss(logits_1.transpose(1, 2), targets)

        # Stage 2 Loss: CE + Smoothing
        loss_2_ce = self.ce_loss(logits_2.transpose(1, 2), targets)
        log_probs_2 = F.log_softmax(logits_2, dim=2)
        loss_2_smooth = self.log_space_smoothing_loss(log_probs_2)
        loss_2 = loss_2_ce + self.smooth_lambda * loss_2_smooth

        # Stage 3 Loss: CE + Smoothing
        loss_3_ce = self.ce_loss(logits_3.transpose(1, 2), targets)
        log_probs_3 = F.log_softmax(logits_3, dim=2)
        loss_3_smooth = self.log_space_smoothing_loss(log_probs_3)
        loss_3 = loss_3_ce + self.smooth_lambda * loss_3_smooth

        # Total Loss
        total_loss = loss_1 + loss_2 + loss_3

        return total_loss


class Trainer:
    """
    Manages training, validation, and model selection.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        self.criterion = CombinedLoss(device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass (returns list of outputs)
            outputs = self.model(features)

            # Compute loss
            loss = self.criterion(outputs, labels)

            # Backward
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        avg_loss = (
            running_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
        )
        print(f"Epoch {epoch} | Train Loss: {avg_loss:.6f}")
        return avg_loss

    def validate(self):
        self.model.eval()
        total_dist = 0
        total_gestures = 0

        with torch.no_grad():
            for features, labels, _ in self.val_loader:
                features = features.to(self.device)
                # labels is (1, T) tensor of frame-wise labels

                # Forward pass
                outputs = self.model(features)
                # Use final stage output for prediction
                final_logits = outputs[-1]  # (1, T, C)

                # Get predictions (Batch size is 1)
                # Squeeze batch dim -> (T, C)
                frame_probs = F.softmax(final_logits.squeeze(0), dim=1).cpu().numpy()

                # Decode predictions
                pred_seq = process_predictions(frame_probs)

                # Decode ground truth
                # Convert tensor to numpy array of IDs
                gt_frame_ids = labels.squeeze(0).cpu().numpy()
                gt_seq = process_predictions(gt_frame_ids)

                # Compute Metric
                dist = compute_levenshtein(pred_seq, gt_seq)
                n_gestures = len(gt_seq)

                total_dist += dist
                total_gestures += n_gestures

        # Avoid division by zero
        if total_gestures == 0:
            score = 0.0
        else:
            score = total_dist / total_gestures

        print(f"Validation Levenshtein Score: {score}")
        return score

    def train(self, num_epochs=Config.NUM_EPOCHS):
        print("Starting training...")

        for epoch in range(1, num_epochs + 1):
            _ = self.train_epoch(epoch)
            val_score = self.validate()

            # Model Selection
            if val_score < self.best_score:
                print(
                    f"New best score: {val_score} (was {self.best_score}). Saving model."
                )
                self.best_score = val_score
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(
                    f"No improvement. Patience: {self.patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Score: {self.best_score}")


def run_training(num_epochs=Config.NUM_EPOCHS):
    # Set seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(Config.SEED)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Data
    train_loader, val_loader, _ = get_dataloaders()

    # Initialize Model
    model = PAKRNet().to(device)

    # Initialize Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # Run Training
    trainer.train(num_epochs=num_epochs)

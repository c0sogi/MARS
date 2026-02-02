import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np

from library.config import Config
from library.data_loader import GestureDataset
from library.model import DGCKN
from library.utils import set_seed, run_length_encoding, compute_metric, setup_logger

# Setup logger
logger = setup_logger("train_module", os.path.join(Config.WORKING_DIR, "train.log"))


class CascadedLoss(nn.Module):
    """
    Cascaded Loss function for DGC-KN.
    Combines Weighted Cross-Entropy and Temporal Smoothing (Log-Space MSE)
    across all three stages of the network.
    Cite solution_lesson_node_00138
    """

    def __init__(self, weight=None):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)
        self.mse_lambda = Config.MSE_LAMBDA

    def forward(self, logits_list, targets):
        total_loss = 0
        C = logits_list[0].shape[2]
        targets_flat = targets.view(-1)

        for logits in logits_list:
            logits_flat = logits.reshape(-1, C)

            # Weighted Cross-Entropy
            loss_ce = self.ce(logits_flat, targets_flat)

            # Temporal Smoothing (Log-Space MSE between adjacent frames)
            # Penalizes rapid changes in prediction confidence
            log_probs = F.log_softmax(logits, dim=2)
            # Truncated MSE with threshold 1.0 (Cite solution_lesson_node_00055)
            loss_smooth = torch.mean(
                torch.clamp(
                    (log_probs[:, 1:, :] - log_probs[:, :-1, :]) ** 2, min=0, max=1.0
                )
            )

            total_loss += loss_ce + self.mse_lambda * loss_smooth

        return total_loss


class Trainer:
    """
    Manages training, validation, and model selection.
    """

    def __init__(
        self, model, train_loader, val_loader, criterion, optimizer, scheduler, device
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for batch in self.train_loader:
            features = batch["features"].to(self.device)
            labels = batch["labels"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass through all stages
            l1, l2, l3 = self.model(features)

            # Compute cascaded loss
            loss = self.criterion([l1, l2, l3], labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                labels = batch["labels"].to(self.device)

                l1, l2, l3 = self.model(features)
                loss = self.criterion([l1, l2, l3], labels)
                total_loss += loss.item()

                # Decode predictions from the final stage (Stage 3)
                # Validation batch size is 1, so we squeeze
                preds_frame = torch.argmax(l3, dim=2).squeeze(0).cpu().numpy()
                targets_frame = labels.squeeze(0).cpu().numpy()

                # Run-Length Encoding for sequence metric
                pred_seq = run_length_encoding(
                    preds_frame, min_length=Config.MIN_GESTURE_LENGTH
                )
                target_seq = run_length_encoding(
                    targets_frame, min_length=Config.MIN_GESTURE_LENGTH
                )

                all_preds.append(pred_seq)
                all_targets.append(target_seq)

        avg_loss = total_loss / len(self.val_loader)
        # Compute Levenshtein distance metric
        score = compute_metric(all_preds, all_targets)
        return avg_loss, score

    def fit(self, epochs):
        logger.info(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_score = self.validate()

            # Log with full precision as requested
            logger.info(
                f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Score (Levenshtein): {val_score}"
            )

            # Scheduler step (ReduceLROnPlateau minimizes metric)
            self.scheduler.step(val_score)

            # Model Selection (Lower Levenshtein score is better)
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)
                logger.info(f"New best model saved with score: {val_score}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    logger.info("Early stopping triggered.")
                    break


def generate_submission():
    """
    Generates predictions for the test set using the best saved model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model structure
    # Input dim: 180 (kinematics) + 13 (audio) = 193
    model = DGCKN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)

    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning("No checkpoint found! Predictions will be random.")

    model.eval()

    # Load Test Data
    test_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "test.csv"),
        mode="test",
        load_cached_data=True,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    results = []
    logger.info("Generating submission...")

    with torch.no_grad():
        for batch in test_loader:
            features = batch["features"].to(device)
            sample_id = batch["id"][0]

            # Forward pass
            _, _, logits3 = model(features)
            preds = torch.argmax(logits3, dim=2).squeeze(0).cpu().numpy()

            # Decode
            pred_seq = run_length_encoding(preds, min_length=Config.MIN_GESTURE_LENGTH)
            pred_str = ",".join(map(str, pred_seq))

            results.append(f"{sample_id},{pred_str}")

    submission_path = os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for line in results:
            f.write(line + "\n")
    logger.info(f"Submission saved to {submission_path}")


def train_model(max_samples=None, epochs=Config.EPOCHS):
    """
    Main entry point for training the model.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Datasets
    train_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "train.csv"),
        mode="train",
        load_cached_data=True,
        max_samples=max_samples,
    )
    val_dataset = GestureDataset(
        os.path.join(Config.METADATA_DIR, "val.csv"),
        mode="val",
        load_cached_data=True,
        max_samples=max_samples,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # Validate on full sequences
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model Initialization
    # Input dim: 180 (kinematics) + 13 (audio) = 193
    model = DGCKN(input_dim=193, num_classes=Config.NUM_CLASSES).to(device)

    # Loss Configuration
    # Class Weighting: Background (0) gets 0.2, others get 1.0
    weights = torch.ones(Config.NUM_CLASSES).to(device)
    weights[0] = Config.BACKGROUND_WEIGHT
    criterion = CascadedLoss(weight=weights)

    # Optimization
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # Start Training
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device
    )
    trainer.fit(epochs)

    # Generate Submission after training
    generate_submission()

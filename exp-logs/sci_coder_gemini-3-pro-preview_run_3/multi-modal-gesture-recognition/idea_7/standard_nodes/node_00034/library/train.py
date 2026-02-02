import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library import config, utils, model, data_loader


class Trainer:
    def __init__(self, device, logger):
        self.device = device
        self.logger = logger

        # Initialize Model
        self.model = model.KC_IRN().to(self.device)

        # Initialize Optimizer (Adam as requested)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.SCHEDULER_PATIENCE,
        )

        # Loss Functions
        # Weighted Focal Loss (Cite Lesson 00010 - Class Weighting evolution)
        class_weights = config.CLASS_WEIGHTS.to(self.device)
        self.criterion_ce = utils.FocalLoss(alpha=class_weights, gamma=2.0)

        # Smoothing Loss (Truncated MSE on log-probs)
        self.criterion_smooth = utils.TruncatedMSELoss(threshold=1.0)

        self.smoothing_lambda = config.SMOOTHING_LAMBDA

    def calculate_loss(self, outputs, targets):
        """
        Computes the cascaded loss: L_total = L1 + L2 + L3
        L2 and L3 include smoothing regularization.
        """
        # Unpack outputs: [log_probs_stage1, log_probs_stage2, log_probs_stage3]
        log_probs_1, log_probs_2, log_probs_3 = outputs

        # Cross Entropy Loss (NLLLoss expects log_probs)
        # Targets shape: (Batch, Time)
        loss_1 = self.criterion_ce(log_probs_1, targets)
        loss_2 = self.criterion_ce(log_probs_2, targets)
        loss_3 = self.criterion_ce(log_probs_3, targets)

        # Smoothing Loss
        smooth_2 = self.criterion_smooth(log_probs_2)
        smooth_3 = self.criterion_smooth(log_probs_3)

        # Total Loss
        total_loss = (
            loss_1
            + (loss_2 + self.smoothing_lambda * smooth_2)
            + (loss_3 + self.smoothing_lambda * smooth_3)
        )

        return total_loss, [loss_1.item(), loss_2.item(), loss_3.item()]

    def train_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (features, labels) in enumerate(loader):
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features)

            # Compute loss
            loss, _ = self.calculate_loss(outputs, labels)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            if hasattr(config, "GRAD_CLIP") and config.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), config.GRAD_CLIP
                )

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(loader)

    def validate(self, loader):
        self.model.eval()

        total_edit_distance = 0
        total_gt_gestures = 0

        with torch.no_grad():
            for features, labels, sample_id in loader:
                features = features.to(self.device)
                # labels in validation loader are full sequence labels

                # Forward pass
                # features shape: (1, Time, InputDim)
                outputs = self.model(features)

                # Use Stage 3 output for final prediction
                log_probs_3 = outputs[2]  # (1, NumClasses, Time)

                # Get frame-wise predictions
                predictions = torch.argmax(log_probs_3, dim=1).squeeze(0).cpu().numpy()

                # Decode to sequence of gesture IDs
                pred_sequence = utils.decode_predictions_to_sequence(
                    predictions, background_id=config.BACKGROUND_CLASS_ID, min_len=5
                )

                # Get Ground Truth sequence
                # labels is a tensor (1, Time), convert to numpy array
                gt_frame_labels = labels.squeeze(0).cpu().numpy()
                gt_sequence = utils.decode_predictions_to_sequence(
                    gt_frame_labels,
                    background_id=config.BACKGROUND_CLASS_ID,
                    min_len=1,  # Don't filter GT too aggressively
                )

                # Calculate Levenshtein Distance
                dist = utils.compute_levenshtein(pred_sequence, gt_sequence)

                total_edit_distance += dist
                total_gt_gestures += len(gt_sequence)

        # Metric: Total Distance / Total GT Gestures
        # Handle division by zero if validation set is empty or has no gestures
        if total_gt_gestures == 0:
            score = 0.0
        else:
            score = total_edit_distance / total_gt_gestures

        return score

    def fit(self, train_loader, val_loader, num_epochs, patience):
        best_score = float("inf")
        patience_counter = 0

        self.logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_score = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step(val_score)

            epoch_time = time.time() - start_time

            self.logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Time: {epoch_time:.2f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Score (Error Rate): {val_score}"
            )

            # Checkpointing & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), config.MODEL_SAVE_PATH)
                self.logger.info(f"New best model saved with score: {best_score}")
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{patience}"
                )

            if patience_counter >= patience:
                self.logger.info("Early stopping triggered.")
                break

        self.logger.info(f"Training complete. Best Validation Score: {best_score}")


def train_model():
    # Set seeds for reproducibility
    utils.set_seed(config.SEED)

    # Setup Logger
    log_path = os.path.join(config.WORKING_DIR, "training.log")
    logger = utils.setup_logger(log_path)

    # Device
    device = config.DEVICE
    logger.info(f"Using device: {device}")

    # Load Data
    logger.info("Loading datasets...")
    train_loader, val_loader, test_loader = data_loader.get_dataloaders(
        load_cached=True
    )

    # Initialize Trainer
    trainer = Trainer(device, logger)

    # Train
    trainer.fit(
        train_loader, val_loader, num_epochs=config.NUM_EPOCHS, patience=config.PATIENCE
    )

    # Generate Submission
    logger.info("Generating submission for test set...")

    # Load best model
    trainer.model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    trainer.model.eval()

    submission_lines = []

    with torch.no_grad():
        for features, _, sample_ids in test_loader:
            features = features.to(device)
            # sample_ids is a tuple of size 1 (batch size 1)
            sample_id = sample_ids[0]

            outputs = trainer.model(features)
            log_probs_3 = outputs[2]

            predictions = torch.argmax(log_probs_3, dim=1).squeeze(0).cpu().numpy()

            pred_sequence = utils.decode_predictions_to_sequence(
                predictions, background_id=config.BACKGROUND_CLASS_ID, min_len=5
            )

            # Format: Id,Sequence
            labels_str = " ".join(map(str, pred_sequence))

            # Sanitize ID (Cite debug_lesson_5)
            clean_id = int("".join(filter(str.isdigit, str(sample_id))))

            line = f"{clean_id},{labels_str}"
            submission_lines.append(line)

    # Save submission
    with open(config.SUBMISSION_PATH, "w") as f:
        f.write("Id,Sequence\n")
        for line in submission_lines:
            f.write(line + "\n")

    logger.info(f"Submission saved to {config.SUBMISSION_PATH}")

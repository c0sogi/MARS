import os
import time
import torch
import torch.optim as optim
import numpy as np
import scipy.signal
from library.config import Config
from library.utils import AverageMeter, compute_levenshtein, setup_logger, set_seed
from library.model import CRGCN
from library.losses import compute_total_loss
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, device, logger):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.device = device
        self.logger = logger
        self.best_score = float("inf")  # Lower Levenshtein score is better

    def _decode_predictions(self, logits, lengths):
        """
        Decodes frame-wise logits into a list of gesture sequences.
        Applies Median Filtering, collapses repeats, and removes background.
        """
        # logits: (B, C, T)
        # lengths: (B,)

        preds = torch.argmax(logits, dim=1).cpu().numpy()  # (B, T)
        decoded_sequences = []

        for i in range(preds.shape[0]):
            length = lengths[i]
            raw_pred = preds[i, :length]

            # Apply Median Filter to smooth predictions
            # Window size must be odd
            window_size = Config.MEDIAN_WINDOW_SIZE
            if window_size % 2 == 0:
                window_size += 1

            # Pad to handle edges if sequence is short
            if len(raw_pred) < window_size:
                smoothed_pred = raw_pred
            else:
                smoothed_pred = scipy.signal.medfilt(raw_pred, kernel_size=window_size)

            # Collapse repeats and remove background (0)
            sequence = []
            prev_label = -1

            for label in smoothed_pred:
                if label != prev_label:
                    if label != 0:  # 0 is background
                        sequence.append(int(label))
                    prev_label = label

            decoded_sequences.append(sequence)

        return decoded_sequences

    def _decode_targets(self, targets, lengths):
        """
        Decodes frame-wise targets into a list of gesture sequences.
        """
        # targets: (B, T)
        targets_np = targets.cpu().numpy()
        decoded_sequences = []

        for i in range(targets_np.shape[0]):
            length = lengths[i]
            raw_target = targets_np[i, :length]

            sequence = []
            prev_label = -1

            for label in raw_target:
                if label != prev_label:
                    if label != 0:
                        sequence.append(int(label))
                    prev_label = label

            decoded_sequences.append(sequence)

        return decoded_sequences

    def train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()

        # Metrics for logging detailed loss components
        stage3_cls_losses = AverageMeter()
        stage3_bnd_losses = AverageMeter()

        for batch_idx, (features, labels, boundaries, mask, lengths) in enumerate(
            self.train_loader
        ):
            features = features.to(self.device)
            labels = labels.to(self.device)
            boundaries = boundaries.to(self.device)
            mask = mask.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute loss
            loss, metrics = compute_total_loss(outputs, labels, boundaries, mask)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIP
            )

            self.optimizer.step()

            # Update metrics
            losses.update(loss.item(), features.size(0))
            if "stage3_cls" in metrics:
                stage3_cls_losses.update(metrics["stage3_cls"], features.size(0))
            if "stage3_bnd" in metrics:
                stage3_bnd_losses.update(metrics["stage3_bnd"], features.size(0))

        self.logger.info(
            f"Epoch [{epoch}/{Config.NUM_EPOCHS}] Train Loss: {losses.avg:.6f} "
            f"(S3 Cls: {stage3_cls_losses.avg:.6f}, S3 Bnd: {stage3_bnd_losses.avg:.6f})"
        )
        return losses.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (features, labels, boundaries, mask, lengths) in enumerate(
                self.val_loader
            ):
                features = features.to(self.device)
                labels = labels.to(self.device)
                boundaries = boundaries.to(self.device)
                mask = mask.to(self.device)

                # Forward pass
                outputs = self.model(features, mask)

                # Compute loss
                loss, _ = compute_total_loss(outputs, labels, boundaries, mask)
                losses.update(loss.item(), features.size(0))

                # Get predictions from Stage 3 (Final Refinement)
                # outputs['stage3'] is (cls_logits, bnd_logits)
                stage3_logits = outputs["stage3"][0]

                # Decode
                batch_preds = self._decode_predictions(stage3_logits, lengths)
                batch_targets = self._decode_targets(labels, lengths)

                all_predictions.extend(batch_preds)
                all_targets.extend(batch_targets)

        # Compute Levenshtein Score
        score = compute_levenshtein(all_predictions, all_targets)

        self.logger.info(f"Validation Loss: {losses.avg:.10f}")
        self.logger.info(f"Validation Levenshtein Error: {score:.10f}")

        return losses.avg, score

    def fit(self):
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_score = self.validate()

            # Checkpoint
            if val_score < self.best_score:
                self.best_score = val_score
                patience_counter = 0
                self.save_checkpoint("best_model.pth")
                self.logger.info(f"New best model found! Score: {self.best_score:.10f}")
            else:
                patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {patience_counter}/{Config.PATIENCE}"
                )

            if patience_counter >= Config.PATIENCE:
                self.logger.info("Early stopping triggered.")
                break

    def save_checkpoint(self, filename):
        path = os.path.join(Config.CHECKPOINT_DIR, filename)
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")


def run_training(debug_size=None, epochs=None):
    """
    Main entry point to setup and run training.
    """
    # Setup
    set_seed(Config.SEED)
    logger = setup_logger("training")

    # Override epochs if provided
    if epochs is not None:
        Config.NUM_EPOCHS = epochs

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Data Loaders
    logger.info("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(debug_size=debug_size)

    # Model
    logger.info("Initializing CRGCN model...")
    model = CRGCN().to(device)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        logger=logger,
    )

    # Start Training
    logger.info("Starting training...")
    start_time = time.time()
    trainer.fit()
    duration = time.time() - start_time
    logger.info(f"Training completed in {duration:.2f} seconds.")

    return trainer.best_score

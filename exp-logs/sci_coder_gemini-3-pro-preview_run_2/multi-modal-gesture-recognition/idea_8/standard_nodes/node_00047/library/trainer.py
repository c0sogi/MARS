import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import nltk
from library import config, utils


class Trainer:
    """
    Trainer class for the DSR-CRCN model.
    Handles training, validation, loss computation, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader):
        """
        Args:
            model (nn.Module): The DSR_CRCN model instance.
            train_loader (DataLoader): DataLoader for training data.
            val_loader (DataLoader): DataLoader for validation data.
        """
        self.device = utils.get_device()
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY,
        )

        # Loss Criteria
        # Weighted Cross Entropy to handle class imbalance (Background vs Gestures)
        class_weights = torch.tensor(config.CLASS_WEIGHTS, dtype=torch.float32).to(
            self.device
        )
        self.ce_criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

    def tmse_loss(self, logits, mask):
        """
        Computes the smoothing loss (T-MSE) on Softmax probabilities.
        Calculates the Mean Squared Error between probabilities of adjacent frames.

        Args:
            logits: [Batch, Time, NumClasses]
            mask: [Batch, Time] (Boolean mask of valid frames)

        Returns:
            Scalar loss tensor.
        """
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=2)  # [B, T, C]

        # Clamp for numerical stability (though softmax output is safe [0,1])
        probs = torch.clamp(probs, min=1e-7, max=1.0)

        # Compute difference between t and t-1
        # diff shape: [B, T-1, C]
        diff = probs[:, 1:, :] - probs[:, :-1, :]

        # Squared Error
        mse = diff**2

        # Determine valid transitions
        # A transition is valid if both t and t-1 are valid frames
        valid_transitions = mask[:, 1:] & mask[:, :-1]  # [B, T-1]

        # Expand mask for broadcasting over classes
        valid_transitions = valid_transitions.unsqueeze(2).float()  # [B, T-1, 1]

        # Compute masked mean
        # Sum of errors over valid transitions
        sum_mse = torch.sum(mse * valid_transitions)

        # Count of valid elements (Batch * ValidTime * Classes)
        count = torch.sum(valid_transitions) * config.NUM_CLASSES

        return sum_mse / (count + 1e-6)

    def compute_combined_loss(self, logits0, logits1, logits2, targets, mask):
        """
        Computes the total loss combining Cross Entropy and T-MSE for all stages.
        """

        # --- Helper for Masked Cross Entropy ---
        def masked_ce(logits_in, targets_in, mask_in):
            # Flatten: [B, T, C] -> [B*T, C]
            flat_logits = logits_in.reshape(-1, config.NUM_CLASSES)
            flat_targets = targets_in.reshape(-1)
            flat_mask = mask_in.reshape(-1).float()

            # Compute element-wise loss
            loss_raw = self.ce_criterion(flat_logits, flat_targets)

            # Apply mask and mean
            loss = (loss_raw * flat_mask).sum() / (flat_mask.sum() + 1e-6)
            return loss

        # --- Stage 0: Generation (Encoder) ---
        # Only Cross Entropy
        loss_gen = masked_ce(logits0, targets, mask)

        # --- Stage 1: Refinement 1 ---
        # Cross Entropy + T-MSE
        loss_ref1_ce = masked_ce(logits1, targets, mask)
        loss_ref1_tmse = self.tmse_loss(logits1, mask)
        loss_ref1 = loss_ref1_ce + config.TMSE_WEIGHT * loss_ref1_tmse

        # --- Stage 2: Refinement 2 ---
        # Cross Entropy + T-MSE
        loss_ref2_ce = masked_ce(logits2, targets, mask)
        loss_ref2_tmse = self.tmse_loss(logits2, mask)
        loss_ref2 = loss_ref2_ce + config.TMSE_WEIGHT * loss_ref2_tmse

        # --- Total Loss ---
        total_loss = (
            config.LAMBDA_GEN * loss_gen
            + config.LAMBDA_REF1 * loss_ref1
            + config.LAMBDA_REF2 * loss_ref2
        )

        metrics = {
            "loss_gen": loss_gen.item(),
            "loss_ref1": loss_ref1.item(),
            "loss_ref2": loss_ref2.item(),
            "tmse1": loss_ref1_tmse.item(),
            "tmse2": loss_ref2_tmse.item(),
        }

        return total_loss, metrics

    def train_epoch(self, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        metrics_accum = {}

        for batch in self.train_loader:
            # Move data to device
            features = batch["features"].to(self.device)
            targets = batch["targets"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits0, logits1, logits2 = self.model(features, mask)

            # Compute loss
            loss, batch_metrics = self.compute_combined_loss(
                logits0, logits1, logits2, targets, mask
            )

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            # Accumulate metrics
            running_loss += loss.item()
            for k, v in batch_metrics.items():
                metrics_accum[k] = metrics_accum.get(k, 0.0) + v

        avg_loss = running_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics_accum.items()}

        return avg_loss, avg_metrics

    def validate(self):
        """
        Runs validation on the validation set.
        Computes the Levenshtein metric instead of just loss.
        Cite Lesson 8: Decoupling Surrogate Loss from Sequence Metrics.
        """
        self.model.eval()

        total_distance = 0
        total_gestures = 0

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                targets = batch["targets"]  # Keep on CPU for decoding
                lengths = batch["lengths"]

                # Forward pass
                _, _, logits2 = self.model(features, mask)

                # Process batch
                for i in range(len(features)):
                    seq_len = lengths[i].item()

                    # Extract logits
                    sample_logits = logits2[i, :seq_len, :]
                    sample_probs = F.softmax(sample_logits, dim=1).cpu().numpy()

                    # Smooth (Cite Lesson 6)
                    smoothed_probs = utils.smooth_predictions(
                        sample_probs, window_size=config.MEDIAN_WINDOW
                    )

                    # Decode
                    pred_seq = utils.decode_sequence(smoothed_probs)

                    # Target
                    target_seq = utils.decode_target_sequence(
                        targets[i, :seq_len].numpy()
                    )

                    # Metric
                    dist = nltk.edit_distance(pred_seq, target_seq)
                    total_distance += dist
                    total_gestures += len(target_seq)

        if total_gestures == 0:
            return 0.0

        return total_distance / total_gestures

    def fit(self):
        """
        Main training loop with Early Stopping based on Levenshtein Metric.
        """
        best_val_metric = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, config.EPOCHS + 1):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate (Metric)
            val_metric = self.validate()

            duration = time.time() - start_time

            # Print metrics
            print(
                f"Epoch {epoch}/{config.EPOCHS} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Metric (Levenshtein): {val_metric:.4f} | "
                f"Time: {duration:.2f}s"
            )

            # Early Stopping & Checkpointing based on Metric (Cite Lesson 8)
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with Val Metric: {val_metric:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                    print(
                        f"Early stopping triggered after {patience_counter} epochs without improvement."
                    )
                    break

        print(f"Training complete. Best Validation Metric: {best_val_metric:.4f}")

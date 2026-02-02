import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import GestureDataset
from library.model import RSKARN
from library.loss import CascadedLoss
from library.utils import run_length_encoding, calculate_score


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        self.device = Config.DEVICE
        self.save_path = Config.MODEL_SAVE_PATH

        # Initialize Model
        self.model = RSKARN().to(self.device)

        # Initialize Loss and Optimizer
        self.criterion = CascadedLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Data Loaders
        self._init_dataloaders()

    def _init_dataloaders(self):
        # Training Data: Windowed samples
        train_dataset = GestureDataset(
            split="train", mode="train", load_cached_data=True
        )
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        # Validation Data: Full sequences for evaluation
        val_dataset = GestureDataset(
            split="val", mode="inference", load_cached_data=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Process one sequence at a time
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Training samples (windows): {len(train_dataset)}")
        print(f"Validation samples (sequences): {len(val_dataset)}")

    def predict_sliding_window(self, features):
        """
        Performs sliding window inference with temporal ensembling.
        Args:
            features: (T, InputDim) tensor
        Returns:
            probs: (T, NumClasses) tensor of averaged probabilities
        """
        self.model.eval()
        T, D = features.shape
        window_size = Config.WINDOW_SIZE
        stride = Config.STRIDE

        # Prepare accumulator for probabilities
        # We only care about Stage 3 output for final prediction
        probs_acc = torch.zeros((T, Config.NUM_CLASSES), device=self.device)
        count_acc = torch.zeros((T, 1), device=self.device)

        # Generate windows
        windows = []
        indices = []

        # Handle case where sequence is shorter than window
        if T < window_size:
            # Pad to window size
            pad_len = window_size - T
            feat_pad = F.pad(features, (0, 0, 0, pad_len))  # Pad time dim
            windows.append(feat_pad)
            indices.append((0, T))
        else:
            for start in range(0, T - window_size + 1, stride):
                end = start + window_size
                windows.append(features[start:end])
                indices.append((start, end))

            # Handle last window if not covered perfectly
            if T > window_size and (T - window_size) % stride != 0:
                start = T - window_size
                end = T
                windows.append(features[start:end])
                indices.append((start, end))

        if not windows:
            return probs_acc

        # Batch processing of windows
        # We can process windows in batches to speed up
        batch_size = Config.BATCH_SIZE

        with torch.no_grad():
            for i in range(0, len(windows), batch_size):
                batch_wins = windows[i : i + batch_size]
                batch_idxs = indices[i : i + batch_size]

                # Stack: (B, Window, Dim)
                input_tensor = torch.stack(batch_wins).to(self.device)

                # Forward
                _, _, s3_logits = self.model(input_tensor)  # (B, C, Window)

                # Softmax
                s3_probs = F.softmax(s3_logits, dim=1)  # (B, C, Window)

                # Accumulate
                for b in range(len(batch_wins)):
                    start, end = batch_idxs[b]
                    # Transpose to (Window, C)
                    p = s3_probs[b].transpose(0, 1)

                    # If we padded, slice valid part
                    valid_len = end - start
                    # Note: indices stored are relative to original T
                    # The window might be padded if T < window_size
                    if T < window_size:
                        p = p[:T]
                        valid_len = T

                    probs_acc[start:end] += p
                    count_acc[start:end] += 1.0

        # Average
        final_probs = probs_acc / (count_acc + 1e-8)
        return final_probs

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        metrics_sum = {}

        for batch_idx, (features, targets) in enumerate(self.train_loader):
            features = features.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            s1, s2, s3 = self.model(features)

            # Loss
            loss, metrics = self.criterion(s1, s2, s3, targets)

            # Backward
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            # Accumulate metrics
            for k, v in metrics.items():
                metrics_sum[k] = metrics_sum.get(k, 0.0) + v

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in metrics_sum.items()}

        return avg_loss, avg_metrics

    def validate(self):
        self.model.eval()

        predictions = {}
        ground_truths = {}

        # Iterate over full sequences
        for features, labels, sample_id in self.val_loader:
            # Features: (1, T, D) -> Squeeze batch dim
            features = features.squeeze(0)
            labels = labels.squeeze(0).numpy()
            sample_id = sample_id[0]

            # Sliding Window Inference
            probs = self.predict_sliding_window(features)  # (T, C)

            # Decode
            pred_labels = torch.argmax(probs, dim=1).cpu().numpy()

            # Run Length Encoding (Dense -> Sparse Gesture IDs)
            pred_seq = run_length_encoding(pred_labels)
            gt_seq = run_length_encoding(labels)

            predictions[sample_id] = pred_seq
            ground_truths[sample_id] = gt_seq

        # Calculate Score
        score = calculate_score(predictions, ground_truths)
        return score

    def fit(self):
        print(f"Starting training on device: {self.device}")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, Config.NUM_EPOCHS + 1):
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)

            # Validate
            val_score = self.validate()

            print(f"Epoch {epoch}/{Config.NUM_EPOCHS}")
            print(f"  Train Loss: {train_loss:.8f}")
            print(f"  Train Metrics: {train_metrics}")
            print(f"  Val Score (Levenshtein Error): {val_score:.8f}")

            # Checkpoint & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), self.save_path)
                print(f"  New best model saved to {self.save_path}")
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Score: {best_score:.8f}")


if __name__ == "__main__":
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Initialize and run trainer
    trainer = Trainer()
    trainer.fit()

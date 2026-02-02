import os
import torch
import torch.optim as optim
import numpy as np
import scipy.signal
from torch.utils.data import DataLoader

from library.config import Config
from library.utils import (
    set_seed,
    save_checkpoint,
    compute_dataset_score,
    compute_levenshtein_distance,
)
from library.model import GHG_CRCN
from library.loss import HierarchicalLoss
from library.data_loader import GestureDataset, collate_fn


class Trainer:
    """
    Trainer class for the GHG-CRCN model.
    Manages training, validation, early stopping, and prediction.
    """

    def __init__(self, device=None):
        """
        Initialize the Trainer.

        Args:
            device (torch.device, optional): Device to run the model on.
                                             If None, automatically selects GPU if available.
        """
        set_seed(Config.SEED)

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = GHG_CRCN().to(self.device)

        # Initialize Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Initialize Loss Function
        self.criterion = HierarchicalLoss().to(self.device)

        # Data Loaders (Lazy initialization in fit method or separate setup)
        self.train_loader = None
        self.val_loader = None

    def _decode_predictions(self, probs_cls, lengths):
        """
        Decodes probability sequences into gesture ID lists.
        Applies Median Filtering and collapses repeats.

        Args:
            probs_cls (torch.Tensor): (Batch, Time, NumClasses)
            lengths (torch.Tensor): (Batch,)

        Returns:
            list[list[int]]: List of predicted gesture sequences.
        """
        predictions = []

        # Move to CPU for numpy operations
        probs_cls = probs_cls.detach().cpu().numpy()
        lengths = lengths.cpu().numpy()

        for i in range(len(lengths)):
            length = lengths[i]
            # Get class indices for the valid sequence length
            # Shape: (Time,)
            seq_probs = probs_cls[i, :length, :]
            seq_preds = np.argmax(seq_probs, axis=1)

            # Apply Median Filter (Label-Space Smoothing)
            # Kernel size must be odd. 7 is a reasonable default for ~10fps data.
            kernel_size = 7
            if len(seq_preds) >= kernel_size:
                # Scipy medfilt doesn't support padding mode directly in the same way we might want,
                # but standard medfilt zero-pads.
                # To implement "Nearest-Neighbor Padding" as per prompt, we can pad manually.
                pad_width = kernel_size // 2
                padded_seq = np.pad(seq_preds, pad_width, mode="edge")
                smoothed_seq = scipy.signal.medfilt(padded_seq, kernel_size)
                # Crop back
                smoothed_seq = smoothed_seq[pad_width:-pad_width]
            else:
                smoothed_seq = seq_preds

            # Collapse repeats and remove background (0)
            decoded_gesture = []
            prev_label = -1

            for label in smoothed_seq:
                label = int(label)
                if label != prev_label:
                    if label != 0:  # 0 is background
                        decoded_gesture.append(label)
                    prev_label = label

            predictions.append(decoded_gesture)

        return predictions

    def train_epoch(self):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in self.train_loader:
            # Move data to device
            features = batch["features"].to(self.device)
            targets_cls = batch["targets_cls"].to(self.device)
            targets_bnd = batch["targets_bnd"].to(self.device)
            targets_fg = batch["targets_fg"].to(self.device)
            mask = batch["mask"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (Deep Supervision: returns list of outputs)
            outputs = self.model(features, mask)

            # Compute Loss
            loss = self.criterion(outputs, targets_cls, targets_bnd, targets_fg, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches if num_batches > 0 else 0.0

    def validate(self):
        """
        Evaluates the model on the validation set.
        Returns the Levenshtein error rate.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in self.val_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]

                # Get ground truth sequences for metric calculation
                # Targets in batch are padded tensors, we need to extract lists
                batch_targets_cls = batch["targets_cls"].cpu().numpy()

                for i in range(len(lengths)):
                    length = lengths[i]
                    # Extract raw target sequence (including 0s, but collapsing logic handles it)
                    # Actually, for Levenshtein, we need the *sequence of gestures*.
                    # The dataset provides frame-wise labels. We must collapse the frame-wise targets
                    # to get the ground truth sequence of gestures.

                    t_seq_frames = batch_targets_cls[i, :length]
                    t_seq_collapsed = []
                    prev = -1
                    for t in t_seq_frames:
                        if t != prev:
                            if t != 0:
                                t_seq_collapsed.append(int(t))
                            prev = t
                    all_targets.append(t_seq_collapsed)

                # Forward pass
                # We only care about the final stage output for inference
                _, _, out3 = self.model(features, mask)

                # Extract Class Probabilities from Stage 3
                # Shape: (Batch, Time, NumClasses + 2) -> take first NumClasses
                probs_cls = out3[:, :, : Config.NUM_CLASSES]

                # Decode
                batch_preds = self._decode_predictions(probs_cls, lengths)
                all_preds.extend(batch_preds)

        # Compute Metric
        score = compute_dataset_score(all_preds, all_targets)
        return score

    def fit(
        self,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
        debug=Config.DEBUG,
    ):
        """
        Main training loop with Early Stopping.
        """
        # Initialize Datasets
        train_dataset = GestureDataset(
            split="train", debug=debug, load_cached_data=True
        )
        val_dataset = GestureDataset(split="val", debug=debug, load_cached_data=True)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,  # Avoid multiprocessing issues in some envs
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        best_score = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            val_score = self.validate()

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Score (Error Rate): {val_score:.10f}"
            )

            # Early Stopping Check
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "best_score": best_score,
                    },
                    filename="best_model.pth",
                )
                # print("New best model saved.")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}.")
                    break

        print(f"Training complete. Best Validation Score: {best_score:.10f}")

    def predict(self, split="test"):
        """
        Generates predictions for a specific dataset split (e.g., 'test').
        Loads the best model checkpoint before predicting.

        Returns:
            dict: {sample_id: [gesture_id, ...]}
        """
        # Load best model
        checkpoint_path = os.path.join(Config.WORKING_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(
                f"Loaded best model from epoch {checkpoint['epoch']} with score {checkpoint['best_score']:.6f}"
            )
        else:
            print("Warning: No checkpoint found. Predicting with current model state.")

        dataset = GestureDataset(split=split, debug=Config.DEBUG, load_cached_data=True)
        loader = DataLoader(
            dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )

        self.model.eval()
        results = {}

        # We need sample IDs to map predictions.
        # The dataset object has metadata.
        # Since shuffle=False, the order matches dataset.metadata.
        sample_ids = dataset.metadata["sample_id"].tolist()
        current_idx = 0

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"]

                _, _, out3 = self.model(features, mask)
                probs_cls = out3[:, :, : Config.NUM_CLASSES]

                batch_preds = self._decode_predictions(probs_cls, lengths)

                for pred in batch_preds:
                    sid = sample_ids[current_idx]
                    results[sid] = pred
                    current_idx += 1

        return results

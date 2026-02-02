import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from library.config import Config
from library.model import MCWINet
from library.data_loader import MultimodalDataset, collate_fn
from library.utils import decode_predictions, compute_levenshtein_ratio, save_submission


class Trainer:
    def __init__(self, device=None):
        """
        Initializes the Trainer with model, optimizer, loss, and scheduler.
        """
        # Set reproducible seeds
        self._set_seed(Config.SEED)

        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize Model
        self.model = MCWINet().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS
        )

        # Loss Function
        # Background weight 0.5, Label Smoothing 0.1
        class_weights = Config.get_class_weights(self.device)
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
        )

        self.best_ler = float("inf")

        # Ensure directories exist
        os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(Config.SUBMISSION_DIR, exist_ok=True)

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def train_epoch(self, train_loader, epoch):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in train_loader:
            if batch is None:
                continue

            # Move data to device
            skeleton = batch["skeleton"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            mask = batch["mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)

            self.optimizer.zero_grad()

            # Forward Pass
            # Logits: (B, T, NumClasses)
            logits = self.model(skeleton, audio, mask, lengths)

            # Flatten for CrossEntropyLoss
            # Logits: (B*T, NumClasses)
            # Labels: (B*T)
            flat_logits = logits.view(-1, Config.NUM_CLASSES)
            flat_labels = labels.view(-1)

            loss = self.criterion(flat_logits, flat_labels)

            # Backward Pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss

    def validate(self, val_loader):
        """
        Runs validation and computes Levenshtein Error Rate.
        """
        self.model.eval()
        hypotheses = []
        references = []

        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)  # Ground Truth
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                # Forward Pass
                logits = self.model(skeleton, audio, mask, lengths)

                # Get Predictions (Frame-wise)
                preds = torch.argmax(logits, dim=2).cpu().numpy()  # (B, T)
                targets = labels.cpu().numpy()  # (B, T)

                batch_lengths = lengths.cpu().numpy()

                for i in range(preds.shape[0]):
                    length = int(batch_lengths[i])

                    # Decode Prediction: Smooth -> RLE -> Filter Background/Short
                    pred_seq = decode_predictions(preds[i][:length])
                    hypotheses.append(pred_seq)

                    # Decode Target: RLE -> Filter Background
                    # We use decode_predictions with min_len=1 to extract the sequence from frame labels
                    target_seq = decode_predictions(targets[i][:length], min_len=1)
                    references.append(target_seq)

        ler = compute_levenshtein_ratio(hypotheses, references)
        return ler

    def fit(self, epochs=Config.NUM_EPOCHS, batch_size=Config.BATCH_SIZE):
        """
        Main training loop.
        """
        print(f"Initializing datasets...")
        train_dataset = MultimodalDataset(mode="train")
        val_dataset = MultimodalDataset(mode="val")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True,
        )

        print(f"Starting training on {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, epoch)
            val_ler = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step()

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.10f} | Val LER: {val_ler:.10f}"
            )

            # Checkpoint
            if val_ler < self.best_ler:
                self.best_ler = val_ler
                self.save_checkpoint()

        print(f"Training complete. Best Val LER: {self.best_ler:.10f}")

    def save_checkpoint(self):
        path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        torch.save(self.model.state_dict(), path)
        # print(f"Saved best model to {path}")

    def predict(
        self, output_path=os.path.join(Config.SUBMISSION_DIR, "submission.csv")
    ):
        """
        Generates predictions for the test set using the best model.
        Uses batch_size=1 to ensure alignment with metadata sample IDs.
        """
        # Load Best Model
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(
                torch.load(checkpoint_path, map_location=self.device)
            )
            print(f"Loaded best model from {checkpoint_path}")
        else:
            print("Warning: No checkpoint found. Using current model weights.")

        self.model.eval()

        test_dataset = MultimodalDataset(mode="test")

        # IMPORTANT: Use batch_size=1 and shuffle=False to preserve order matching test_dataset.df
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=True,
        )

        predictions = []

        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    # If a sample is corrupt/empty, we append empty prediction to maintain count
                    predictions.append([])
                    continue

                skeleton = batch["skeleton"].to(self.device)
                audio = batch["audio"].to(self.device)
                mask = batch["mask"].to(self.device)
                lengths = batch["lengths"].to(self.device)

                logits = self.model(skeleton, audio, mask, lengths)
                preds = torch.argmax(logits, dim=2).cpu().numpy()  # (1, T)

                length = int(lengths[0].item())
                pred_seq = decode_predictions(preds[0][:length])
                predictions.append(pred_seq)

        # Align with Sample IDs
        sample_ids = test_dataset.df["sample_id"].tolist()

        if len(sample_ids) != len(predictions):
            print(
                f"Warning: Mismatch in samples ({len(sample_ids)}) and predictions ({len(predictions)})."
            )
            # Truncate or pad to match (though batch_size=1 should prevent this unless collate drops items)
            min_len = min(len(sample_ids), len(predictions))
            sample_ids = sample_ids[:min_len]
            predictions = predictions[:min_len]

        save_submission(sample_ids, predictions, output_path)
        print(f"Submission saved to {output_path}")

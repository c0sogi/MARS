import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm

from library.config import (
    NUM_CLASSES,
    LEARNING_RATE,
    WEIGHT_DECAY,
    BATCH_SIZE,
    NUM_EPOCHS,
    PATIENCE,
    BG_CLASS_WEIGHT,
    LABEL_SMOOTHING,
    WORKING_DIR,
    SEED,
    GRAD_CLIP,
    DEBUG,
)
from library.model import AGGRN
from library.data_loader import get_dataloaders, set_seed
from library.utils import (
    calculate_levenshtein_accuracy,
    post_process_output,
    rle_decode,
    compute_levenshtein,
)


class Trainer:
    def __init__(
        self, model, device, train_loader, val_loader, learning_rate=LEARNING_RATE
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        # Define Loss with Class Weights
        # Class 0 is Background, 1-20 are Gestures
        weights = torch.ones(NUM_CLASSES).to(device)
        weights[0] = BG_CLASS_WEIGHT

        self.criterion = nn.CrossEntropyLoss(
            weight=weights, label_smoothing=LABEL_SMOOTHING, reduction="mean"
        )

        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0.0

        for batch in self.train_loader:
            # Unpack batch
            skeleton, audio, labels, mask, lengths = batch

            # Move to device
            skeleton = skeleton.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)
            # lengths stays on CPU for pack_padded_sequence usually, or move if needed by model logic
            # In our model, lengths is used for pack_padded_sequence which expects CPU tensor in older pytorch,
            # but newer versions handle it. We'll keep it as is or move to cpu if it was on gpu.

            self.optimizer.zero_grad()

            # Forward Pass
            # logits: (B, T, NUM_CLASSES)
            logits = self.model(skeleton, audio, lengths)

            # Flatten for loss calculation
            # We only calculate loss on valid frames (indicated by mask)
            # mask: (B, T) boolean
            active_logits = logits[mask]
            active_labels = labels[mask]

            loss = self.criterion(active_logits, active_labels)

            # Backward
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        total_loss = 0.0
        total_dist = 0
        total_ref_len = 0

        with torch.no_grad():
            for batch in self.val_loader:
                skeleton, audio, labels, mask, lengths = batch

                skeleton = skeleton.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                logits = self.model(skeleton, audio, lengths)

                # Loss calculation
                active_logits = logits[mask]
                active_labels = labels[mask]
                loss = self.criterion(active_logits, active_labels)
                total_loss += loss.item()

                # Metric calculation (Levenshtein)
                # Apply Softmax
                probs = torch.softmax(logits, dim=2)

                # Iterate over batch to decode sequences
                for i in range(len(labels)):
                    # Get valid length for this sequence
                    length = lengths[i]

                    # Slice to valid length
                    seq_probs = probs[i, :length, :].cpu().numpy()
                    seq_labels = labels[i, :length].cpu().numpy()

                    # Decode Predictions
                    pred_seq = post_process_output(
                        seq_probs, window_size=5, min_len=5, bg_class=0
                    )

                    # Decode Targets (Ground Truth)
                    # We use rle_decode on GT as well to get the sequence of IDs,
                    # assuming GT is clean but frame-wise.
                    target_seq = rle_decode(seq_labels, bg_class=0, min_len=1)

                    dist = compute_levenshtein(pred_seq, target_seq)
                    total_dist += dist
                    total_ref_len += len(target_seq)

        avg_loss = total_loss / len(self.val_loader)

        # Avoid division by zero
        ler = total_dist / total_ref_len if total_ref_len > 0 else 0.0

        return avg_loss, ler

    def fit(self, num_epochs=NUM_EPOCHS, patience=PATIENCE):
        best_ler = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(WORKING_DIR, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_ler = self.validate()

            self.scheduler.step(val_ler)

            print(
                f"Epoch {epoch}/{num_epochs} - "
                f"Train Loss: {train_loss:.6f} - "
                f"Val Loss: {val_loss:.6f} - "
                f"Val LER: {val_ler:.6f}"
            )

            # Checkpointing based on LER (Metric)
            if val_ler < best_ler:
                best_ler = val_ler
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                # print(f"  New best model saved! LER: {best_ler:.6f}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

        print(f"Training complete. Best Val LER: {best_ler:.6f}")
        return best_ler


def train_model(load_cached_data=True, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Main function to setup and run training.

    Args:
        load_cached_data (bool): Whether to use cached preprocessed data.
        num_epochs (int): Number of training epochs.
        batch_size (int): Batch size for data loaders.
    """
    # Set seeds for reproducibility
    set_seed(SEED)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Loaders
    # Note: The data_loader module handles caching internally based on existence,
    # but we can force re-creation if needed by clearing cache manually or modifying loader.
    # The provided data_loader class accepts `load_cached_data` in __init__ but
    # get_dataloaders doesn't expose it directly in the provided snippet.
    # We assume the provided get_dataloaders uses the default behavior or we instantiate datasets manually.
    # Based on provided file content, get_dataloaders doesn't take load_cached_data arg,
    # but MultimodalDataset does. We will instantiate manually to respect the argument.

    from library.data_loader import MultimodalDataset, pad_collate
    from torch.utils.data import DataLoader

    train_ds = MultimodalDataset(split="train", load_cached_data=load_cached_data)
    val_ds = MultimodalDataset(split="val", load_cached_data=load_cached_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=pad_collate,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=pad_collate,
    )

    # Model
    model = AGGRN()

    # Trainer
    trainer = Trainer(model, device, train_loader, val_loader)

    # Run Training
    trainer.fit(num_epochs=num_epochs, patience=PATIENCE)

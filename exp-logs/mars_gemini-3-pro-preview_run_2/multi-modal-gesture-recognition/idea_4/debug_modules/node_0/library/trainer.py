import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from library.utils import (
    seed_everything,
    smooth_predictions,
    decode_sequence,
    compute_normalized_levenshtein,
)
from library.data_loader import GestureDataset, collate_fn
from library.model import TemporalTransformer


class Trainer:
    def __init__(self, config, device=None):
        """
        Args:
            config (dict): Configuration dictionary containing hyperparameters.
            device (torch.device): Device to run training on.
        """
        self.config = config
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Model initialization
        self.model = TemporalTransformer(
            input_dim=config.get("input_dim", 85),
            num_classes=config.get("num_classes", 21),
            d_model=config.get("d_model", 128),
            nhead=config.get("nhead", 4),
            num_layers=config.get("num_layers", 2),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
        ).to(self.device)

        # Loss Function: Weighted Cross Entropy
        # Class 0 is background (weight 0.1), others are 1.0
        weights = torch.ones(config.get("num_classes", 21))
        weights[0] = 0.1
        self.criterion = nn.CrossEntropyLoss(weight=weights, reduction="none").to(
            self.device
        )

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.get("learning_rate", 1e-4),
            weight_decay=config.get("weight_decay", 1e-4),
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        self.working_dir = "./working/idea_4"
        os.makedirs(self.working_dir, exist_ok=True)
        self.checkpoint_path = os.path.join(self.working_dir, "best_model.pth")

    def train_epoch(self, dataloader, noise_std=0.01):
        self.model.train()
        total_loss = 0.0
        total_frames = 0

        for batch_idx, (features, labels, lengths, mask) in enumerate(dataloader):
            features = features.to(self.device)
            labels = labels.to(self.device)
            mask = mask.to(self.device)

            # Apply Gaussian Noise for regularization
            if noise_std > 0:
                noise = torch.randn_like(features) * noise_std
                features = features + noise

            self.optimizer.zero_grad()

            # Forward pass
            # src_key_padding_mask in nn.Transformer expects True for padded positions
            # Our mask has 1 for valid, 0 for padding. So padding_mask = ~mask
            padding_mask = ~mask
            outputs = self.model(
                features, src_key_padding_mask=padding_mask
            )  # (B, T, C)

            # Flatten for loss calculation
            # outputs: (B * T, C)
            # labels: (B * T)
            outputs_flat = outputs.view(-1, outputs.size(-1))
            labels_flat = labels.view(-1)
            mask_flat = mask.view(-1)

            # Compute loss per element
            loss_unreduced = self.criterion(outputs_flat, labels_flat)

            # Apply mask to ignore padding (although padding is 0 and has weight,
            # we strictly want to ignore padding indices, not just background class)
            masked_loss = loss_unreduced * mask_flat.float()

            loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * mask_flat.sum().item()
            total_frames += mask_flat.sum().item()

        return total_loss / (total_frames + 1e-8)

    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0.0
        total_frames = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_idx, (features, labels, lengths, mask) in enumerate(dataloader):
                features = features.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                padding_mask = ~mask
                outputs = self.model(features, src_key_padding_mask=padding_mask)

                # Loss calculation
                outputs_flat = outputs.view(-1, outputs.size(-1))
                labels_flat = labels.view(-1)
                mask_flat = mask.view(-1)

                loss_unreduced = self.criterion(outputs_flat, labels_flat)
                masked_loss = loss_unreduced * mask_flat.float()
                loss = masked_loss.sum() / (mask_flat.sum() + 1e-8)

                total_loss += loss.item() * mask_flat.sum().item()
                total_frames += mask_flat.sum().item()

                # Decoding for Levenshtein metric
                probs = torch.softmax(outputs, dim=2)
                preds = torch.argmax(probs, dim=2)  # (B, T)

                # Convert to CPU list for processing
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                for i in range(len(lengths_np)):
                    length = lengths_np[i]
                    # Extract valid sequence
                    p_seq = preds_np[i, :length]
                    t_seq = labels_np[i, :length]

                    # 1. Smooth
                    p_smooth = smooth_predictions(p_seq, window_size=5)

                    # 2. Decode
                    p_decoded = decode_sequence(p_smooth, background_class_id=0)
                    t_decoded = decode_sequence(t_seq, background_class_id=0)

                    all_preds.append(p_decoded)
                    all_targets.append(t_decoded)

        avg_loss = total_loss / (total_frames + 1e-8)
        levenshtein_score = compute_normalized_levenshtein(all_preds, all_targets)

        return avg_loss, levenshtein_score

    def fit(self, train_loader, val_loader, epochs=50, patience=10):
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {epochs} epochs on device {self.device}...")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(
                train_loader, noise_std=self.config.get("noise_std", 0.01)
            )
            val_loss, val_lev = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.10f} | "
                f"Val Levenshtein: {val_lev:.10f}"
            )

            # Early Stopping based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), self.checkpoint_path)
                # print(f"Model saved at epoch {epoch}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        print("Training complete.")

    def predict(self, test_loader, output_file="./submission/submission.csv"):
        # Load best model
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(
                torch.load(self.checkpoint_path, map_location=self.device)
            )
            print(f"Loaded best model from {self.checkpoint_path}")
        else:
            print("Warning: No checkpoint found, using current model weights.")

        self.model.eval()
        results = []

        # Map back to sample IDs requires access to dataset or loader structure
        # We assume the loader iterates in order of the dataset
        sample_ids = test_loader.dataset.sample_ids
        current_idx = 0

        with torch.no_grad():
            for batch_idx, (features, labels, lengths, mask) in enumerate(test_loader):
                features = features.to(self.device)
                mask = mask.to(self.device)
                padding_mask = ~mask

                outputs = self.model(features, src_key_padding_mask=padding_mask)
                preds = torch.argmax(outputs, dim=2)  # (B, T)

                preds_np = preds.cpu().numpy()
                lengths_np = lengths.cpu().numpy()

                batch_size = len(lengths_np)

                for i in range(batch_size):
                    length = lengths_np[i]
                    p_seq = preds_np[i, :length]

                    # Smooth and Decode
                    p_smooth = smooth_predictions(p_seq, window_size=5)
                    p_decoded = decode_sequence(p_smooth, background_class_id=0)

                    seq_str = "".join(["," + str(x) for x in p_decoded])
                    # Format: SessionID,2,12,3 (comma separated, first is ID)
                    # Actually prompt says: Session00001,2,12,3
                    # p_decoded is list of ints.

                    sid = sample_ids[current_idx]

                    # If empty sequence
                    if not p_decoded:
                        line = f"{sid}"
                    else:
                        line = f"{sid}," + ",".join(map(str, p_decoded))

                    results.append(line)
                    current_idx += 1

        # Save submission
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w") as f:
            for line in results:
                f.write(line + "\n")
        print(f"Submission saved to {output_file}")


def run_training_pipeline():
    seed_everything(42)

    # Configuration
    config = {
        "input_dim": 85,  # 72 (Skeleton) + 13 (Audio)
        "num_classes": 21,  # 20 Gestures + 1 Background
        "d_model": 128,
        "nhead": 4,
        "num_layers": 2,
        "dim_feedforward": 512,
        "dropout": 0.1,
        "learning_rate": 1e-3,
        "weight_decay": 1e-4,
        "batch_size": 32,
        "epochs": 50,
        "patience": 10,
        "noise_std": 0.05,  # Gaussian noise std for regularization
    }

    # Data Loading
    train_dataset = GestureDataset(
        metadata_file="./metadata/train.csv", load_cached_data=True
    )
    val_dataset = GestureDataset(
        metadata_file="./metadata/val.csv", load_cached_data=True
    )
    test_dataset = GestureDataset(
        metadata_file="./metadata/test.csv", load_cached_data=True, is_test=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
    )

    # Trainer
    trainer = Trainer(config)

    # Fit
    trainer.fit(
        train_loader, val_loader, epochs=config["epochs"], patience=config["patience"]
    )

    # Predict
    trainer.predict(test_loader, output_file="./submission/submission.csv")

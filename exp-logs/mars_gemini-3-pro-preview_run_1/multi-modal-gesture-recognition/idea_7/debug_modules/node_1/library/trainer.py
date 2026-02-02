import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from library.config import (
    BEST_MODEL_PATH,
    SUBMISSION_PATH,
    NUM_CLASSES,
    BACKGROUND_LABEL,
    BACKGROUND_WEIGHT,
    LABEL_SMOOTHING,
    LEARNING_RATE,
    WEIGHT_DECAY,
    LAMBDA_BOUNDARY,
    NUM_EPOCHS,
    BATCH_SIZE,
    SEED,
    WORKING_DIR,
)
from library.utils import set_seed, levenshtein_distance, smooth_predictions, rle_decode
from library.model import KAGRN
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self, device=None):
        self.device = (
            device
            if device
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        set_seed(SEED)

        # Initialize Model
        self.model = KAGRN().to(self.device)

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
        )

        # Loss Functions
        # Class weights: Downweight background
        class_weights = torch.ones(NUM_CLASSES).to(self.device)
        class_weights[BACKGROUND_LABEL] = BACKGROUND_WEIGHT

        self.criterion_cls = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=LABEL_SMOOTHING
        )
        self.criterion_bnd = nn.BCEWithLogitsLoss()

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # Move data to device
            pos = batch["pos"].to(self.device)
            vel = batch["vel"].to(self.device)
            audio = batch["audio"].to(self.device)
            labels = batch["labels"].to(self.device)
            boundaries = batch["boundaries"].to(self.device)
            lengths = batch[
                "lengths"
            ]  # CPU tensor is fine for pack_padded_sequence usually, but model handles it

            # Forward Pass
            cls_logits, bnd_logits = self.model(pos, vel, audio, lengths)

            # Flatten for Loss Calculation
            # cls_logits: (B, T, C) -> (B*T, C)
            # labels: (B, T) -> (B*T)
            # Masking padding is implicitly handled if we ignore index,
            # but CrossEntropyLoss with ignore_index or careful masking is better.
            # Here, we rely on the fact that padded labels are BACKGROUND_LABEL (0),
            # and we have a weight for it. However, to be precise, we should mask.
            # Since we use pack_padded_sequence in RNN, the output is padded with zeros/garbage after length.
            # Let's construct a mask based on lengths to be safe.

            mask = (
                torch.arange(labels.size(1), device=self.device)[None, :]
                < lengths.to(self.device)[:, None]
            )
            mask = mask.view(-1)

            cls_flat = cls_logits.view(-1, NUM_CLASSES)[mask]
            lbl_flat = labels.view(-1)[mask]

            bnd_flat = bnd_logits.view(-1)[mask]
            bnd_lbl_flat = boundaries.view(-1)[mask]

            # Calculate Loss
            loss_cls = self.criterion_cls(cls_flat, lbl_flat)
            loss_bnd = self.criterion_bnd(bnd_flat, bnd_lbl_flat)

            loss = loss_cls + LAMBDA_BOUNDARY * loss_bnd

            # Backward Pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        total_dist = 0
        total_gestures = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                pos = batch["pos"].to(self.device)
                vel = batch["vel"].to(self.device)
                audio = batch["audio"].to(self.device)
                labels = batch["labels"].to(self.device)
                boundaries = batch["boundaries"].to(self.device)
                lengths = batch["lengths"]

                cls_logits, bnd_logits = self.model(pos, vel, audio, lengths)

                # Loss Calculation (for monitoring)
                mask = (
                    torch.arange(labels.size(1), device=self.device)[None, :]
                    < lengths.to(self.device)[:, None]
                )
                mask = mask.view(-1)
                cls_flat = cls_logits.view(-1, NUM_CLASSES)[mask]
                lbl_flat = labels.view(-1)[mask]
                bnd_flat = bnd_logits.view(-1)[mask]
                bnd_lbl_flat = boundaries.view(-1)[mask]

                loss_cls = self.criterion_cls(cls_flat, lbl_flat)
                loss_bnd = self.criterion_bnd(bnd_flat, bnd_lbl_flat)
                loss = loss_cls + LAMBDA_BOUNDARY * loss_bnd
                total_loss += loss.item()

                # Decoding for Metric
                probs = torch.softmax(cls_logits, dim=2)
                preds = torch.argmax(probs, dim=2).cpu().numpy()
                targets = labels.cpu().numpy()

                for i in range(len(lengths)):
                    length = int(lengths[i].item())
                    # Extract valid sequence
                    p_seq = preds[i, :length]
                    t_seq = targets[i, :length]

                    # Post-processing
                    p_seq_smooth = smooth_predictions(p_seq)

                    # RLE Decode
                    hyp = rle_decode(p_seq_smooth)
                    ref = rle_decode(t_seq)

                    # Metric
                    dist = levenshtein_distance(hyp, ref)
                    total_dist += dist
                    total_gestures += len(ref)

        avg_loss = total_loss / len(loader)
        # Avoid division by zero
        error_rate = total_dist / total_gestures if total_gestures > 0 else 1.0

        return avg_loss, error_rate

    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS, patience=10):
        print(f"Starting training on {self.device} for {epochs} epochs.")
        best_score = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_score = self.evaluate(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"Time: {elapsed:.1f}s | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein Rate: {val_score:.6f}"
            )

            # Checkpointing & Early Stopping
            if val_score < best_score:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), BEST_MODEL_PATH)
                print(f"  >>> New Best Model Saved (Score: {best_score:.6f})")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {epoch} epochs.")
                    break

        print(f"Training complete. Best Validation Score: {best_score:.6f}")

    def predict(self, test_loader):
        print("Loading best model for inference...")
        if not os.path.exists(BEST_MODEL_PATH):
            print("Warning: Best model not found. Using current model state.")
        else:
            self.model.load_state_dict(
                torch.load(BEST_MODEL_PATH, map_location=self.device)
            )

        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                pos = batch["pos"].to(self.device)
                vel = batch["vel"].to(self.device)
                audio = batch["audio"].to(self.device)
                lengths = batch["lengths"]
                sample_ids = batch["sample_ids"]

                cls_logits, _ = self.model(pos, vel, audio, lengths)

                probs = torch.softmax(cls_logits, dim=2)
                preds = torch.argmax(probs, dim=2).cpu().numpy()

                for i in range(len(sample_ids)):
                    length = int(lengths[i].item())
                    sid = sample_ids[i]

                    # Extract and process
                    p_seq = preds[i, :length]
                    p_seq_smooth = smooth_predictions(p_seq)
                    hyp = rle_decode(p_seq_smooth)

                    # Format string: "ID,1,2,3"
                    hyp_str = ",".join(map(str, hyp))
                    results.append(f"{sid},{hyp_str}")

        # Save Submission
        with open(SUBMISSION_PATH, "w") as f:
            for line in results:
                f.write(f"{line}\n")

        print(f"Submission saved to {SUBMISSION_PATH}")


def run_training_pipeline():
    # Ensure working directory exists
    os.makedirs(WORKING_DIR, exist_ok=True)

    # Load Data
    train_loader, val_loader, test_loader = get_dataloaders()

    # Initialize Trainer
    trainer = Trainer()

    # Train
    trainer.fit(train_loader, val_loader, epochs=NUM_EPOCHS)

    # Predict
    trainer.predict(test_loader)

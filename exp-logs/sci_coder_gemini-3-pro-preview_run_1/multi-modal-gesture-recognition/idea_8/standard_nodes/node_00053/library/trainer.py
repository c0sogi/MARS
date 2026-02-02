import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, rle_decode, compute_levenshtein_ratio
from library.model import KAMTRN
from library.data_loader import get_dataloaders


class Trainer:
    def __init__(self):
        self.device = Config.get_device()
        print(f"Using device: {self.device}")

        # Initialize Model
        self.model = KAMTRN().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS, eta_min=1e-6
        )

        # Loss Functions
        # Class weights: Downweight background (index 0)
        class_weights = torch.ones(Config.NUM_CLASSES).to(self.device)
        class_weights[0] = Config.BACKGROUND_WEIGHT

        self.criterion_cls = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=Config.LABEL_SMOOTHING
        )
        self.criterion_bnd = nn.BCEWithLogitsLoss()

        # Early Stopping State
        self.best_score = float("inf")
        self.patience_counter = 0

    def train_epoch(self, loader):
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            # Unpack batch
            pose, velocity, audio, labels, boundaries, lengths = batch

            # Move to device
            pose = pose.to(self.device)
            velocity = velocity.to(self.device)
            audio = audio.to(self.device)
            labels = labels.to(self.device)
            boundaries = boundaries.to(self.device)

            # Forward Pass
            class_logits, boundary_logits = self.model(pose, velocity, audio)

            # Reshape for Loss Calculation
            # Flatten batch and time dimensions
            # class_logits: (B, T, C) -> (B*T, C)
            # labels: (B, T) -> (B*T)
            B, T, C = class_logits.shape
            flat_class_logits = class_logits.reshape(-1, C)
            flat_labels = labels.reshape(-1)

            # boundary_logits: (B, T, 1) -> (B, T)
            flat_boundary_logits = boundary_logits.squeeze(-1)

            # Compute Losses
            loss_cls = self.criterion_cls(flat_class_logits, flat_labels)
            loss_bnd = self.criterion_bnd(flat_boundary_logits, boundaries)

            # Multi-Task Combination
            loss = loss_cls + Config.BOUNDARY_LOSS_WEIGHT * loss_bnd

            # Optimization Step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), Config.GRADIENT_CLIP_VAL
            )
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        self.model.eval()
        total_loss = 0.0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                pose, velocity, audio, labels, boundaries, lengths = batch

                pose = pose.to(self.device)
                velocity = velocity.to(self.device)
                audio = audio.to(self.device)
                labels = labels.to(self.device)
                boundaries = boundaries.to(self.device)

                # Forward
                class_logits, boundary_logits = self.model(pose, velocity, audio)

                # Loss Calculation (for monitoring)
                B, T, C = class_logits.shape
                loss_cls = self.criterion_cls(
                    class_logits.reshape(-1, C), labels.reshape(-1)
                )
                loss_bnd = self.criterion_bnd(boundary_logits.squeeze(-1), boundaries)
                loss = loss_cls + Config.BOUNDARY_LOSS_WEIGHT * loss_bnd
                total_loss += loss.item()

                # Decoding for Metric
                probs = torch.softmax(class_logits, dim=2).cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Process each sequence in the batch
                for i in range(B):
                    length = lengths[i]

                    # Slice valid sequence (ignore padding)
                    valid_probs = probs[i, :length, :]
                    valid_targets = labels_np[i, :length]

                    # Decode Predictions
                    pred_seq = rle_decode(
                        valid_probs,
                        min_length=Config.MIN_SEGMENT_LENGTH,
                        background_class=0,
                    )
                    all_preds.append(pred_seq)

                    # Decode Targets (Ground Truth)
                    # Use min_length=1 to capture all annotated GT gestures
                    target_seq = rle_decode(
                        valid_targets, min_length=1, background_class=0
                    )
                    all_targets.append(target_seq)

        # Compute Levenshtein Error Rate
        score = compute_levenshtein_ratio(all_preds, all_targets)

        return score, total_loss / len(loader)

    def train(self):
        print("Starting training...")
        train_loader, val_loader, _ = get_dataloaders()

        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

        for epoch in range(Config.NUM_EPOCHS):
            train_loss = self.train_epoch(train_loader)
            val_score, val_loss = self.validate(val_loader)

            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            print(
                f"Epoch {epoch+1}/{Config.NUM_EPOCHS} | "
                f"Train Loss: {train_loss:.6f} | "
                f"Val Loss: {val_loss:.6f} | "
                f"Val Levenshtein Error: {val_score:.10f} | "
                f"LR: {current_lr:.6f}"
            )

            # Checkpointing based on Levenshtein Error
            if val_score < self.best_score:
                self.best_score = val_score
                self.patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
                print(f"New best model saved with score: {val_score:.10f}")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        print(f"Training finished. Best Validation Score: {self.best_score:.10f}")

    def predict(self):
        print("Generating predictions for test set...")
        _, _, test_loader = get_dataloaders()

        # Load Best Model
        best_model_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
        if not os.path.exists(best_model_path):
            print(
                "No checkpoint found! Running inference with untrained model (not recommended)."
            )
        else:
            self.model.load_state_dict(
                torch.load(best_model_path, map_location=self.device)
            )
            print("Loaded best model checkpoint.")

        self.model.eval()
        predictions = []

        # Get sample IDs from test metadata to ensure order
        test_df = pd.read_csv(Config.TEST_CSV)
        sample_ids = test_df["sample_id"].tolist()

        idx_counter = 0

        with torch.no_grad():
            for batch in test_loader:
                pose, velocity, audio, _, _, lengths = batch

                pose = pose.to(self.device)
                velocity = velocity.to(self.device)
                audio = audio.to(self.device)

                class_logits, _ = self.model(pose, velocity, audio)
                probs = torch.softmax(class_logits, dim=2).cpu().numpy()

                # Batch size is 1 for test loader
                length = lengths[0]
                valid_probs = probs[0, :length, :]

                # Decode
                pred_seq = rle_decode(
                    valid_probs,
                    min_length=Config.MIN_SEGMENT_LENGTH,
                    background_class=0,
                )

                # Format string: "Label1,Label2,..."
                pred_str = ",".join(map(str, pred_seq))

                # Store with ID
                current_id = sample_ids[idx_counter]
                predictions.append((current_id, pred_str))
                idx_counter += 1

        # Write Submission
        with open(Config.SUBMISSION_PATH, "w") as f:
            for sid, pstr in predictions:
                f.write(f"{sid},{pstr}\n")

        print(f"Submission saved to {Config.SUBMISSION_PATH}")


def run():
    set_seed(Config.SEED)
    trainer = Trainer()
    trainer.train()
    trainer.predict()


if __name__ == "__main__":
    # This block is for local testing only, the main execution is expected via import or external script
    run()

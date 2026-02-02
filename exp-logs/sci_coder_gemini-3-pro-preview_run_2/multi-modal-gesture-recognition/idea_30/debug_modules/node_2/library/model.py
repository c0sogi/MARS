import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from library.config import (
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    NUM_STAGES,
    DROPOUT,
    DILATIONS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    NUM_EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    EARLY_STOPPING_MIN_DELTA,
    CHECKPOINT_DIR,
    SUBMISSION_DIR,
    RANDOM_SEED,
    MEDIAN_FILTER_KERNEL,
)
from library.layers import MCAGCN
from library.losses import DeepSupervisionLoss
from library.data_loader import get_dataloaders
from library.utils import set_seed, levenshtein_score, post_process_predictions


class GestureRecognitionModel:
    """
    Wrapper class for the MCAG-CN model to handle training, validation, and inference.
    """

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize the architecture defined in library/layers.py
        self.model = MCAGCN(
            input_dim=INPUT_DIM,
            hidden_dim=HIDDEN_DIM,
            num_classes=NUM_CLASSES,
            num_stages=NUM_STAGES,
            dropout=DROPOUT,
            dilations=DILATIONS,
        ).to(self.device)

        # Initialize the multi-task loss defined in library/losses.py
        self.criterion = DeepSupervisionLoss().to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

    def train_epoch(self, loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        total_loss = 0.0

        for batch in loader:
            features = batch["features"].to(self.device)
            target_cls = batch["target_cls"].to(self.device)
            target_bnd = batch["target_bnd"].to(self.device)
            mask = batch["mask"].to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(features, mask)

            # Compute deep supervision loss
            loss, _ = self.criterion(outputs, target_cls, target_bnd, mask)

            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(loader)

    def validate(self, loader):
        """
        Evaluates the model on the validation set using Levenshtein distance.
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in loader:
                features = batch["features"].to(self.device)
                target_cls = batch["target_cls"].to(self.device)
                target_bnd = batch["target_bnd"].to(self.device)
                mask = batch["mask"].to(self.device)

                outputs = self.model(features, mask)
                loss, _ = self.criterion(outputs, target_cls, target_bnd, mask)
                total_loss += loss.item()

                # Use the output from the final refinement stage
                final_stage = outputs[-1]
                cls_probs = final_stage["cls_probs"]  # (B, T, C)

                # Process batch for metrics
                lengths = mask.sum(dim=1).long().cpu().numpy()
                cls_probs_np = cls_probs.cpu().numpy()
                targets_np = target_cls.cpu().numpy()

                for i in range(len(features)):
                    length = lengths[i]

                    # Get sequence for this sample
                    probs = cls_probs_np[i, :length, :]

                    # Post-process to get predicted gesture IDs
                    pred_seq = post_process_predictions(
                        probs, filter_kernel=MEDIAN_FILTER_KERNEL
                    )
                    all_preds.append(pred_seq)

                    # Construct ground truth sequence from frame-wise targets
                    # Collapse duplicates and remove background (0)
                    t_seq_frames = targets_np[i, :length]
                    t_seq_list = []
                    last_t = None
                    for t in t_seq_frames:
                        if t != last_t:
                            if t != 0:
                                t_seq_list.append(int(t))
                            last_t = t
                    all_targets.append(t_seq_list)

        avg_loss = total_loss / len(loader)
        score = levenshtein_score(all_preds, all_targets)

        return avg_loss, score

    def fit(self, train_loader, val_loader, epochs=NUM_EPOCHS):
        """
        Main training loop with early stopping.
        """
        best_score = float("inf")
        patience_counter = 0
        best_model_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss, val_score = self.validate(val_loader)

            print(
                f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | Val Levenshtein: {val_score:.6f}"
            )

            # Check for improvement
            if val_score < best_score - EARLY_STOPPING_MIN_DELTA:
                best_score = val_score
                patience_counter = 0
                torch.save(self.model.state_dict(), best_model_path)
            else:
                patience_counter += 1

            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

        print(f"Training finished. Best Validation Score: {best_score:.6f}")

        # Restore best model
        if os.path.exists(best_model_path):
            self.model.load_state_dict(torch.load(best_model_path))

    def predict(self, test_loader):
        """
        Generates predictions for the test set in the submission format.
        """
        self.model.eval()
        results = []

        with torch.no_grad():
            for batch in test_loader:
                features = batch["features"].to(self.device)
                mask = batch["mask"].to(self.device)
                sample_ids = batch["sample_ids"]

                outputs = self.model(features, mask)
                final_stage = outputs[-1]
                cls_probs = final_stage["cls_probs"]

                lengths = mask.sum(dim=1).long().cpu().numpy()
                cls_probs_np = cls_probs.cpu().numpy()

                for i in range(len(features)):
                    length = lengths[i]
                    probs = cls_probs_np[i, :length, :]
                    s_id = sample_ids[i]

                    # Post-process
                    pred_seq = post_process_predictions(
                        probs, filter_kernel=MEDIAN_FILTER_KERNEL
                    )

                    # Format: SessionID,label1,label2...
                    pred_str = ",".join(map(str, pred_seq))
                    results.append(f"{s_id},{pred_str}")

        return results


def run_pipeline(epochs=NUM_EPOCHS, batch_size=BATCH_SIZE):
    """
    Orchestrates the data loading, training, and submission generation.
    """
    set_seed(RANDOM_SEED)

    # 1. Load Data
    print("Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=batch_size, load_cached=True
    )

    # 2. Initialize Model
    trainer = GestureRecognitionModel()

    # 3. Train
    trainer.fit(train_loader, val_loader, epochs=epochs)

    # 4. Predict
    print("Generating predictions...")
    predictions = trainer.predict(test_loader)

    # 5. Save Submission
    submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")
    with open(submission_path, "w") as f:
        for line in predictions:
            f.write(line + "\n")

    print(f"Submission saved to {submission_path}")

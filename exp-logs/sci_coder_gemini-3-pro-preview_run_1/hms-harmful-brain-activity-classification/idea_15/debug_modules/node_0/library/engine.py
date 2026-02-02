import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from library.config import Config
from library.utils import AverageMeter, kl_divergence_score, softmax_and_normalize
from library.models import DeepSupervisedModel


class Trainer:
    """
    Manages the training, validation, and inference lifecycle for the
    Deeply-Supervised Coordinate-Fusion Network.
    """

    def __init__(self, model, device, train_loader, val_loader, optimizer, scheduler):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler

        # KL Divergence Loss for training (expects log-probs as input)
        self.criterion = nn.KLDivLoss(reduction="batchmean")

        self.best_val_score = float("inf")

    def train_one_epoch(self, epoch):
        """
        Runs one epoch of training with multi-task supervision.
        """
        self.model.train()
        loss_meter = AverageMeter()

        # Progress bar
        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]",
            leave=False,
        )

        for batch in pbar:
            # Unpack batch
            # X_eeg: (B, 20, T), X_spec: (B, 5, H, W), y: (B, 6)
            X_eeg, X_spec, y = batch

            X_eeg = X_eeg.to(self.device, non_blocking=True)
            X_spec = X_spec.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass (returns 3 logits)
            joint_logit, eeg_logit, spec_logit = self.model(X_eeg, X_spec)

            # Compute Multi-Task Loss
            # KLDivLoss expects log_softmax input
            loss_joint = self.criterion(F.log_softmax(joint_logit, dim=1), y)
            loss_eeg = self.criterion(F.log_softmax(eeg_logit, dim=1), y)
            loss_spec = self.criterion(F.log_softmax(spec_logit, dim=1), y)

            total_loss = (
                Config.LOSS_WEIGHT_JOINT * loss_joint
                + Config.LOSS_WEIGHT_EEG * loss_eeg
                + Config.LOSS_WEIGHT_SPEC * loss_spec
            )

            # Backward pass
            total_loss.backward()

            # Gradient clipping (stability)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Step scheduler (OneCycleLR steps per batch)
            if self.scheduler is not None:
                self.scheduler.step()

            # Update metrics
            loss_meter.update(total_loss.item(), X_eeg.size(0))
            pbar.set_postfix(loss=f"{loss_meter.avg:.4f}")

        return loss_meter.avg

    def validate(self):
        """
        Evaluates the model on the validation set using the Joint Head.
        Returns the average KL Divergence score.
        """
        self.model.eval()
        score_meter = AverageMeter()

        pbar = tqdm(self.val_loader, desc="Validating", leave=False)

        with torch.no_grad():
            for batch in pbar:
                X_eeg, X_spec, y = batch

                X_eeg = X_eeg.to(self.device, non_blocking=True)
                X_spec = X_spec.to(self.device, non_blocking=True)
                y_true = y.cpu().numpy()

                # Forward pass - only need joint logits for validation
                joint_logit, _, _ = self.model(X_eeg, X_spec)

                # Convert logits to probabilities
                y_pred = softmax_and_normalize(joint_logit.cpu().numpy())

                # Compute Metric
                score = kl_divergence_score(y_pred, y_true)
                score_meter.update(score, X_eeg.size(0))

        return score_meter.avg

    def fit(self):
        """
        Main training loop.
        """
        print(f"Starting training for {Config.EPOCHS} epochs on device: {self.device}")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_score = self.validate()

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.10f} | Val KL: {val_score:.10f}"
            )

            # Checkpoint
            if val_score < self.best_val_score:
                print(
                    f"Validation score improved ({self.best_val_score:.6f} --> {val_score:.6f}). Saving model..."
                )
                self.best_val_score = val_score
                save_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save(self.model.state_dict(), save_path)

            # Garbage collection
            gc.collect()
            torch.cuda.empty_cache()


def predict_and_submit(test_loader, model, device):
    """
    Generates predictions for the test set and saves submission.csv.
    """
    print("Loading best model for inference...")
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")

    if not os.path.exists(checkpoint_path):
        print(
            f"Warning: Checkpoint not found at {checkpoint_path}. Using current model weights."
        )
    else:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)

    all_preds = []

    # Inference Loop
    pbar = tqdm(test_loader, desc="Inference")
    with torch.no_grad():
        for batch in pbar:
            # Test loader returns (X_eeg, X_spec)
            X_eeg, X_spec = batch

            X_eeg = X_eeg.to(device)
            X_spec = X_spec.to(device)

            # Forward
            joint_logit, _, _ = model(X_eeg, X_spec)

            # Probabilities
            probs = softmax_and_normalize(joint_logit.cpu().numpy())
            all_preds.append(probs)

    # Concatenate all predictions
    predictions = np.vstack(all_preds)

    # Prepare Submission DataFrame
    # Load test metadata to get eeg_ids
    test_df = pd.read_csv(Config.TEST_CSV)

    # Ensure lengths match
    if len(predictions) != len(test_df):
        print(
            f"Warning: Prediction count {len(predictions)} != Test ID count {len(test_df)}"
        )

    submission = pd.DataFrame(predictions, columns=Config.OUTPUT_COLS)
    submission.insert(0, "eeg_id", test_df["eeg_id"])

    # Save
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")
    print("Head of submission:")
    print(submission.head())

import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from library.config import Config
from library.utils import seed_everything, get_device, compute_metric
from library.data import prepare_data, get_feature_columns, engineer_features
from library.model import WideProjectedNet

# =========================================================================
# Loss Function
# =========================================================================


class MaskedHybridLoss(nn.Module):
    """
    Computes the weighted sum of L1 losses for the final and auxiliary heads.
    Loss is masked to calculate error only during the inspiratory phase (u_out == 0).
    Formula: Loss = L1_final + aux_weight * L1_aux
    """

    def __init__(self, aux_weight=Config.AUX_LOSS_WEIGHT):
        super().__init__()
        self.aux_weight = aux_weight
        self.l1 = nn.L1Loss(reduction="none")

    def forward(self, final_pred, aux_pred, target, u_out):
        """
        Args:
            final_pred: (Batch, Seq, 1) or (Batch, Seq)
            aux_pred: (Batch, Seq, 1) or (Batch, Seq) or None
            target: (Batch, Seq)
            u_out: (Batch, Seq) - Raw binary control input (0=Inspiratory, 1=Expiratory)
        """
        # Flatten all tensors
        final_pred = final_pred.reshape(-1)
        target = target.reshape(-1)
        u_out = u_out.reshape(-1)

        # Create Mask: 1 for Inspiratory (u_out == 0), 0 for Expiratory
        # Using 1 - u_out works because u_out is 0 or 1.
        mask = 1.0 - u_out
        mask_sum = mask.sum() + 1e-8  # Avoid division by zero

        # Final Head Loss
        loss_final = (self.l1(final_pred, target) * mask).sum() / mask_sum

        total_loss = loss_final

        # Auxiliary Head Loss
        if aux_pred is not None:
            aux_pred = aux_pred.reshape(-1)
            loss_aux = (self.l1(aux_pred, target) * mask).sum() / mask_sum
            total_loss += self.aux_weight * loss_aux

        return total_loss


# =========================================================================
# Trainer
# =========================================================================


class Trainer:
    def __init__(self, model, device, train_loader, val_loader):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.criterion = MaskedHybridLoss(aux_weight=Config.AUX_LOSS_WEIGHT)

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # OneCycleLR Scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=Config.LEARNING_RATE,
            steps_per_epoch=len(self.train_loader),
            epochs=Config.EPOCHS,
            pct_start=0.3,
            anneal_strategy="cos",
        )

        self.best_mae = float("inf")
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for batch_idx, (X, u_out, y) in enumerate(self.train_loader):
            X, u_out, y = X.to(self.device), u_out.to(self.device), y.to(self.device)

            self.optimizer.zero_grad()

            # Forward
            final_pred, aux_pred = self.model(X)

            # Loss
            loss = self.criterion(final_pred, aux_pred, y, u_out)

            # Backward
            loss.backward()

            # Gradient Clipping
            clip_grad_norm_(self.model.parameters(), Config.GRAD_CLIP)

            # Update
            self.optimizer.step()
            self.scheduler.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(self.train_loader)
        return avg_loss

    def validate(self):
        self.model.eval()
        preds = []
        targets = []
        u_outs = []

        with torch.no_grad():
            for X, u_out, y in self.val_loader:
                X = X.to(self.device)

                # Forward (discard aux head for validation metric)
                final_pred, _ = self.model(X)

                preds.append(final_pred.cpu())
                targets.append(y)
                u_outs.append(u_out)

        preds = torch.cat(preds)
        targets = torch.cat(targets)
        u_outs = torch.cat(u_outs)

        mae = compute_metric(preds, targets, u_outs)
        return mae

    def fit(self):
        print(f"Starting training for {Config.EPOCHS} epochs...")

        for epoch in range(1, Config.EPOCHS + 1):
            train_loss = self.train_epoch(epoch)
            val_mae = self.validate()

            print(
                f"Epoch {epoch}/{Config.EPOCHS} | Train Loss: {train_loss:.6f} | Val MAE: {val_mae:.6f}"
            )

            # Checkpointing
            if val_mae < self.best_mae:
                self.best_mae = val_mae
                self.patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(f"  -> Model saved! Best MAE: {self.best_mae:.6f}")
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= Config.PATIENCE:
                print(f"Early stopping triggered after {epoch} epochs.")
                break


# =========================================================================
# Helper Functions
# =========================================================================


def get_feature_names():
    """
    Determines the feature names by loading a small sample of the training data
    and applying the feature engineering pipeline.
    """
    # Load enough rows for a few breaths (80 steps per breath) to ensure diffs/lags work
    sample_size = Config.SEQ_LEN * 5
    df = pd.read_csv(Config.TRAIN_PATH, nrows=sample_size)
    df = engineer_features(df)
    features = get_feature_columns(df)
    return features


def generate_submission(model, test_loader, test_ids, device):
    """
    Generates predictions for the test set and saves to submission.csv.
    """
    print("Generating submission...")
    model.eval()
    all_preds = []

    with torch.no_grad():
        for X, u_out in test_loader:
            X = X.to(device)
            final_pred, _ = model(X)
            # Flatten predictions to match (Batch * Seq,)
            all_preds.append(final_pred.cpu().reshape(-1))

    # Concatenate all predictions
    predictions = torch.cat(all_preds).numpy()

    # Ensure lengths match
    if len(predictions) != len(test_ids):
        print(
            f"Warning: Prediction length {len(predictions)} != ID length {len(test_ids)}"
        )

    # Create DataFrame
    sub_df = pd.DataFrame({"id": test_ids, "pressure": predictions})

    # Save
    sub_df.to_csv(Config.SUBMISSION_FILE_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_FILE_PATH}")


# =========================================================================
# Main Execution
# =========================================================================


def main():
    # 1. Setup
    Config.setup()
    seed_everything(Config.SEED)
    device = get_device()

    # 2. Data Preparation
    print("Preparing data...")
    train_loader, val_loader, test_loader, test_ids = prepare_data()

    # 3. Determine Input Dimension and Feature Names
    # We need feature names to identify context columns (R, C, etc.) for the model
    feature_names = get_feature_names()
    input_dim = len(feature_names)
    print(f"Input Dimension: {input_dim}")
    print(f"Feature Names: {feature_names}")

    # 4. Model Initialization
    print("Initializing model...")
    model = WideProjectedNet(input_dim=input_dim, feature_names=feature_names)
    model = model.to(device)

    # 5. Training
    trainer = Trainer(model, device, train_loader, val_loader)
    trainer.fit()

    # 6. Inference
    # Load best model
    print("Loading best model for inference...")
    model.load_state_dict(torch.load(Config.MODEL_SAVE_PATH, map_location=device))

    generate_submission(model, test_loader, test_ids, device)

    print("Done!")


# Execute main
if __name__ == "__main__":
    main()

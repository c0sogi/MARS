import os
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from library.config import (
    EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    MODEL_SAVE_PATH,
    SUBMISSION_PATH,
    IDX_TO_LABEL,
    ID_COL,
    TARGET_COL,
    INPUT_DIM,
    HIDDEN_DIM,
    NUM_CLASSES,
    SEED,
)
from library.utils import seed_everything, accuracy_score, AverageMeter, get_device
from library.data import get_dataloaders
from library.model import DeepVectorDCNResNet


class Trainer:
    """
    Manages the training, validation, and inference process.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Optimization
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
        )

        # Scheduler: Cosine Annealing decaying to 0 over EPOCHS
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=EPOCHS, eta_min=0.0
        )

        self.criterion = nn.CrossEntropyLoss()
        self.best_model_state = None
        self.best_val_acc = -1.0

    def train_one_epoch(self, epoch_idx):
        self.model.train()
        losses = AverageMeter()
        accuracies = AverageMeter()

        for batch_X, batch_y in self.train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(batch_X)
            loss = self.criterion(logits, batch_y)

            loss.backward()
            self.optimizer.step()

            acc = accuracy_score(logits, batch_y)
            losses.update(loss.item(), batch_X.size(0))
            accuracies.update(acc, batch_X.size(0))

        return losses.avg, accuracies.avg

    def validate(self):
        self.model.eval()
        losses = AverageMeter()
        accuracies = AverageMeter()

        with torch.no_grad():
            for batch_X, batch_y in self.val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)

                logits = self.model(batch_X)
                loss = self.criterion(logits, batch_y)
                acc = accuracy_score(logits, batch_y)

                losses.update(loss.item(), batch_X.size(0))
                accuracies.update(acc, batch_X.size(0))

        return losses.avg, accuracies.avg

    def fit(self, epochs=EPOCHS, patience=10):
        print(f"Starting training on device: {self.device}")
        print(f"Epochs: {epochs}, Batch Size: {self.train_loader.batch_size}")

        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            train_loss, train_acc = self.train_one_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Step the scheduler
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.scheduler.step()

            duration = time.time() - start_time

            print(
                f"Epoch {epoch}/{epochs} | "
                f"LR: {current_lr:.2e} | "
                f"Train Loss: {train_loss:.6f} | Train Acc: {train_acc:.6f} | "
                f"Val Loss: {val_loss:.6f} | Val Acc: {val_acc:.6f} | "
                f"Time: {duration:.2f}s"
            )

            # Checkpointing with deepcopy
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
                # Save to disk immediately as a backup
                torch.save(self.best_model_state, MODEL_SAVE_PATH)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(
                    f"Early stopping triggered after {epoch} epochs. Best Val Acc: {self.best_val_acc:.6f}"
                )
                break

        print("Training complete.")
        # Load best model for inference
        if self.best_model_state is not None:
            print(f"Restoring best model with Val Acc: {self.best_val_acc:.6f}")
            self.model.load_state_dict(self.best_model_state)

    def predict(self, test_loader, test_ids):
        self.model.eval()
        predictions = []

        print("Generating predictions...")
        with torch.no_grad():
            for batch_X in test_loader:
                batch_X = batch_X.to(self.device)
                logits = self.model(batch_X)
                # Get class indices
                _, preds = torch.max(logits, dim=1)
                predictions.extend(preds.cpu().numpy())

        # Map indices back to original labels
        final_preds = [IDX_TO_LABEL[idx] for idx in predictions]

        # Create submission dataframe
        df_sub = pd.DataFrame({ID_COL: test_ids, TARGET_COL: final_preds})

        return df_sub


def run_training():
    # 1. Setup
    seed_everything(SEED)
    device = get_device()

    # 2. Data
    print("Initializing DataLoaders...")
    train_loader, val_loader, test_loader, test_ids = get_dataloaders()

    # 3. Model
    print("Initializing Model...")
    model = DeepVectorDCNResNet(
        input_dim=INPUT_DIM, hidden_dim=HIDDEN_DIM, num_classes=NUM_CLASSES
    )

    # 4. Trainer
    trainer = Trainer(model, train_loader, val_loader, device)

    # 5. Training
    # We use a patience of 15 to allow the Cosine Annealing to work its magic
    # without stopping too early during local fluctuations.
    trainer.fit(epochs=EPOCHS, patience=15)

    # 6. Inference
    df_submission = trainer.predict(test_loader, test_ids)

    # 7. Save Submission
    print(f"Saving submission to {SUBMISSION_PATH}...")
    df_submission.to_csv(SUBMISSION_PATH, index=False)
    print("Done.")

import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast

from library.config import Config
from library.utils import (
    set_seed,
    get_device,
    save_checkpoint,
    load_checkpoint,
    optimize_f1_threshold,
)
from library.model import DilatedWideAndDeep
from library.data_processing import process_data, get_dataloaders


class Trainer:
    """
    Encapsulates the training, validation, and prediction logic.
    """

    def __init__(self, model, optimizer, scheduler, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.scaler = GradScaler()

    def train_epoch(self, train_loader):
        """
        Trains the model for one epoch using Automatic Mixed Precision (AMP).
        """
        self.model.train()
        total_loss = 0.0

        for tokens, labels in train_loader:
            tokens = tokens.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            with autocast():
                logits = self.model(tokens)
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        """
        Evaluates the model on the validation set.
        Returns average loss, F1 score, and the optimal threshold.
        """
        self.model.eval()
        total_loss = 0.0
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for tokens, labels in val_loader:
                tokens = tokens.to(self.device)
                labels = labels.to(self.device)

                with autocast():
                    logits = self.model(tokens)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)

        # Concatenate all batches
        y_probs = np.concatenate(all_probs)
        y_true = np.concatenate(all_targets)

        # Calculate metrics
        best_thr, f1 = optimize_f1_threshold(y_true, y_probs)

        return avg_loss, f1, best_thr

    def predict(self, test_loader):
        """
        Generates predictions for the test set.
        """
        self.model.eval()
        all_probs = []

        with torch.no_grad():
            for tokens in test_loader:
                tokens = tokens.to(self.device)
                with autocast():
                    logits = self.model(tokens)
                probs = torch.sigmoid(logits)
                all_probs.append(probs.cpu().numpy())

        return np.concatenate(all_probs)


def run_pipeline(load_cached_data=True):
    """
    Main execution function.
    1. Loads/Processes Data
    2. Initializes Model & Optimizer
    3. Runs Training Loop with Early Stopping
    4. Generates Submission
    """
    set_seed(Config.SEED)
    device = get_device()

    # 1. Data Processing
    print("Initializing Data Processing...")
    (
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        test_ids,
        tokenizer_handler,
        target_encoder,
    ) = process_data(load_cached_data=load_cached_data)

    # 2. Create DataLoaders
    print("Creating DataLoaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        train_tokens,
        train_labels,
        val_tokens,
        val_labels,
        test_tokens,
        batch_size=Config.BATCH_SIZE,
    )

    # 3. Initialize Model
    print("Initializing Model...")
    num_classes = train_labels.shape[1]
    model = DilatedWideAndDeep(
        vocab_size=Config.VOCAB_SIZE,
        num_classes=num_classes,
        embed_dim=Config.EMBED_DIM,
        num_filters=Config.NUM_FILTERS,
        kernel_size=Config.KERNEL_SIZE,
        dilation_rates=Config.DILATION_RATES,
        dropout=Config.DROPOUT,
    ).to(device)

    # 4. Setup Optimization
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)

    # OneCycleLR
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        steps_per_epoch=steps_per_epoch,
        epochs=Config.EPOCHS,
    )

    criterion = nn.BCEWithLogitsLoss()

    # 5. Initialize Trainer
    trainer = Trainer(model, optimizer, scheduler, criterion, device)

    # 6. Training Loop
    print(f"Starting Training for {Config.EPOCHS} epochs...")
    best_f1 = 0.0
    best_thr = 0.5
    patience_counter = 0

    for epoch in range(1, Config.EPOCHS + 1):
        t0 = time.time()

        train_loss = trainer.train_epoch(train_loader)
        val_loss, val_f1, thr = trainer.validate(val_loader)

        dt = time.time() - t0
        print(
            f"Epoch {epoch} | Time: {dt}s | Train Loss: {train_loss} | Val Loss: {val_loss} | Val F1: {val_f1} | Best Thr: {thr}"
        )

        # Checkpointing & Early Stopping
        if val_f1 > best_f1:
            best_f1 = val_f1
            best_thr = thr
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch, val_f1, Config.MODEL_SAVE_PATH)
            print(f"  -> New Best Model Saved! F1: {best_f1}")
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(
                    f"Early stopping triggered after {patience_counter} epochs without improvement."
                )
                break

    # 7. Inference
    print("Starting Inference...")
    # Load best model weights
    checkpoint = load_checkpoint(Config.MODEL_SAVE_PATH, model, device=device)
    print(
        f"Loaded best model from epoch {checkpoint['epoch']} (F1: {checkpoint['score']})"
    )

    # Predict probabilities
    probs = trainer.predict(test_loader)

    # 8. Generate Submission
    print("Generating Submission CSV...")
    # Apply optimized threshold
    binary_preds = (probs >= best_thr).astype(int)

    # Convert binary vectors back to tag strings
    pred_tags_list = target_encoder.inverse_transform(binary_preds)
    pred_tags_str = [" ".join(tags) for tags in pred_tags_list]

    submission = pd.DataFrame({"Id": test_ids, "Tags": pred_tags_str})

    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

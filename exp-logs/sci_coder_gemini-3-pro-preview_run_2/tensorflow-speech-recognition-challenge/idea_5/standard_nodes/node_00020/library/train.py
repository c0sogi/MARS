import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from tqdm import tqdm

from library.config import Config
from library.utils import set_seed, save_checkpoint, load_checkpoint
from library.dataset import get_dataloaders
from library.model import SwinAudioClassifier


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, patience=5):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.patience = patience

        # Optimization components
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.NUM_EPOCHS
        )

        self.best_val_acc = -1.0
        self.patience_counter = 0

    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # DataLoader returns (features, labels, fnames)
        for batch_idx, (inputs, targets, _) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets, _ in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total
        return val_loss, val_acc

    def fit(self, num_epochs=Config.NUM_EPOCHS):
        print(f"Starting training for {num_epochs} epochs on {self.device}...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update scheduler
            self.scheduler.step()

            epoch_time = time.time() - start_time

            print(f"Epoch {epoch+1}/{num_epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {train_loss} | Train Acc: {train_acc}")
            print(f"  Val Loss:   {val_loss} | Val Acc:   {val_acc}")

            # Early Stopping and Checkpointing
            if val_acc > self.best_val_acc:
                print(
                    f"  [Improvement] Val Acc increased from {self.best_val_acc} to {val_acc}. Saving model..."
                )
                self.best_val_acc = val_acc
                self.patience_counter = 0

                # Save best model
                state = {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "best_val_acc": self.best_val_acc,
                }
                save_checkpoint(state, Config.CHECKPOINT_PATH)
            else:
                self.patience_counter += 1
                print(
                    f"  [No Improvement] Patience: {self.patience_counter}/{self.patience}"
                )

            if self.patience_counter >= self.patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Validation Accuracy: {self.best_val_acc}")


def generate_submission(model, test_loader, device, output_path):
    print("Generating submission...")
    model.eval()

    predictions = []
    fnames_list = []

    # Mapping from index to label string
    idx_to_label = {i: label for i, label in enumerate(Config.LABELS)}

    with torch.no_grad():
        for inputs, _, fnames in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            _, preds = outputs.max(1)

            predictions.extend(preds.cpu().numpy())
            fnames_list.extend(fnames)

    # Convert indices to labels
    predicted_labels = [idx_to_label[idx] for idx in predictions]

    # Create DataFrame
    df_submission = pd.DataFrame({"fname": fnames_list, "label": predicted_labels})

    # Save to CSV
    df_submission.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")
    print(f"Submission shape: {df_submission.shape}")
    print(df_submission.head())


def main(debug_subset_size=Config.DEBUG_SUBSET_SIZE):
    # 1. Setup
    set_seed(Config.SEED)
    device = torch.device(Config.DEVICE)
    print(f"Using device: {device}")

    # 2. Data
    # get_dataloaders handles caching internally
    train_loader, val_loader, test_loader = get_dataloaders(
        debug_subset_size=debug_subset_size,
        load_cached_data=True,
        batch_size=Config.BATCH_SIZE,
        num_workers=Config.NUM_WORKERS,
    )

    # 3. Model
    model = SwinAudioClassifier(num_classes=Config.NUM_CLASSES)
    model.to(device)

    # 4. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        patience=5,  # Early stopping patience
    )

    trainer.fit(num_epochs=Config.NUM_EPOCHS)

    # 5. Inference / Submission
    # Load best model weights
    print("Loading best model for inference...")
    try:
        load_checkpoint(Config.CHECKPOINT_PATH, model, device=Config.DEVICE)
    except FileNotFoundError:
        print(
            "Warning: Checkpoint not found. Using current model weights (likely from last epoch)."
        )

    generate_submission(model, test_loader, device, Config.SUBMISSION_PATH)

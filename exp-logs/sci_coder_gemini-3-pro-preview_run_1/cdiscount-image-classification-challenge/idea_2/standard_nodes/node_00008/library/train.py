import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.dataset import BSONProductDataset, collate_fn
from library.model import HierarchicalResNet50


def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Trainer:
    """
    Manages the training and validation lifecycle of the Hierarchical ResNet-50.
    """

    def __init__(self, model, train_loader, val_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss Functions
        # Target (Level 3): Use Label Smoothing
        self.criterion_l3 = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
        # Auxiliary (Level 1 & 2): Standard Cross Entropy
        self.criterion_aux = nn.CrossEntropyLoss()

    def train_epoch(self, optimizer, scheduler, epoch_idx):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct_l3 = 0
        total_samples = 0

        start_time = time.time()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            # Images: (B, Num_Images, C, H, W)
            images = batch["images"].to(self.device)

            # Labels
            targets_l3 = batch["labels"]["target"].to(self.device)
            targets_l1 = batch["labels"]["l1"].to(self.device)
            targets_l2 = batch["labels"]["l2"].to(self.device)

            optimizer.zero_grad()

            # Forward Pass
            outputs = self.model(images)

            # Compute Losses
            loss_l3 = self.criterion_l3(outputs["target"], targets_l3)
            loss_l2 = self.criterion_aux(outputs["l2"], targets_l2)
            loss_l1 = self.criterion_aux(outputs["l1"], targets_l1)

            # Weighted Sum
            total_loss = (
                loss_l3 + (Config.LAMBDA_L2 * loss_l2) + (Config.LAMBDA_L1 * loss_l1)
            )

            # Backward Pass
            total_loss.backward()
            optimizer.step()
            scheduler.step()

            # Metrics
            running_loss += total_loss.item() * images.size(0)

            # Track L3 Accuracy (Target)
            _, preds = torch.max(outputs["target"], 1)
            correct_l3 += torch.sum(preds == targets_l3).item()
            total_samples += images.size(0)

        epoch_loss = running_loss / total_samples
        epoch_acc = correct_l3 / total_samples
        duration = time.time() - start_time

        print(
            f"Epoch {epoch_idx} Training: Loss={epoch_loss}, L3_Accuracy={epoch_acc}, Time={duration}s"
        )
        return epoch_loss, epoch_acc

    def validate(self):
        """Runs validation on the validation set."""
        self.model.eval()
        running_loss = 0.0

        correct_l1 = 0
        correct_l2 = 0
        correct_l3 = 0
        total_samples = 0

        with torch.no_grad():
            for batch in self.val_loader:
                images = batch["images"].to(self.device)

                targets_l3 = batch["labels"]["target"].to(self.device)
                targets_l1 = batch["labels"]["l1"].to(self.device)
                targets_l2 = batch["labels"]["l2"].to(self.device)

                outputs = self.model(images)

                loss_l3 = self.criterion_l3(outputs["target"], targets_l3)
                loss_l2 = self.criterion_aux(outputs["l2"], targets_l2)
                loss_l1 = self.criterion_aux(outputs["l1"], targets_l1)

                total_loss = (
                    loss_l3
                    + (Config.LAMBDA_L2 * loss_l2)
                    + (Config.LAMBDA_L1 * loss_l1)
                )

                running_loss += total_loss.item() * images.size(0)

                # Calculate accuracies for all levels
                _, preds_l1 = torch.max(outputs["l1"], 1)
                correct_l1 += torch.sum(preds_l1 == targets_l1).item()

                _, preds_l2 = torch.max(outputs["l2"], 1)
                correct_l2 += torch.sum(preds_l2 == targets_l2).item()

                _, preds_l3 = torch.max(outputs["target"], 1)
                correct_l3 += torch.sum(preds_l3 == targets_l3).item()

                total_samples += images.size(0)

        val_loss = running_loss / total_samples
        acc_l1 = correct_l1 / total_samples
        acc_l2 = correct_l2 / total_samples
        acc_l3 = correct_l3 / total_samples

        print(f"Validation: Loss={val_loss}")
        print(f"Validation Accuracies: L1={acc_l1}, L2={acc_l2}, L3={acc_l3}")

        return val_loss, acc_l3


def train_model():
    """
    Main function to setup and run the training pipeline.
    """
    set_seed(Config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Datasets and Loaders
    print("Initializing Datasets...")
    train_dataset = BSONProductDataset(mode="train")
    val_dataset = BSONProductDataset(mode="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # 2. Model
    print("Initializing Model...")
    model = HierarchicalResNet50()
    model.to(device)

    # 3. Optimizer and Scheduler
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY
    )

    # OneCycleLR Scheduler
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=Config.LEARNING_RATE,
        epochs=Config.EPOCHS,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.3,  # Warmup for first 30%
        div_factor=25.0,
        final_div_factor=1000.0,
    )

    # 4. Training Loop
    trainer = Trainer(model, train_loader, val_loader, device)

    best_val_acc = 0.0
    patience = 3
    patience_counter = 0

    print("Starting Training...")
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch + 1}/{Config.EPOCHS}")

        # Train
        train_loss, train_acc = trainer.train_epoch(optimizer, scheduler, epoch + 1)

        # Validate
        val_loss, val_acc_l3 = trainer.validate()

        # Checkpointing
        if val_acc_l3 > best_val_acc:
            print(
                f"New best L3 accuracy: {val_acc_l3} (was {best_val_acc}). Saving model..."
            )
            best_val_acc = val_acc_l3
            torch.save(model.state_dict(), Config.MODEL_CHECKPOINT)
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")

        # Early Stopping
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

    print(f"Training Complete. Best Validation L3 Accuracy: {best_val_acc}")

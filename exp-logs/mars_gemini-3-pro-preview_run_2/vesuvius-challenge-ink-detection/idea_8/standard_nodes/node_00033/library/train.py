import os
import random
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library import config
from library import utils
from library import model
from library import dataset


def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class Trainer:
    def __init__(self, device=config.DEVICE):
        self.device = device
        self.working_dir = config.WORKING_DIR
        os.makedirs(self.working_dir, exist_ok=True)

        self.model = model.InkSegFormer(pretrained=config.PRETRAINED).to(self.device)
        self.criterion = utils.BCEDiceLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3
        )

        self.best_score = 0.0
        self.validation_threshold = config.VALIDATION_THRESHOLD

    def train_one_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0

        for batch in dataloader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def validate(self, dataloader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["image"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                # Collect for metric calculation
                # We keep them on CPU to avoid OOM during concatenation if validation set is huge,
                # but for F0.5 calculation on batches, we can do it batch-wise or accumulate.
                # Given the metric function works on tensors, we'll accumulate tensors.
                all_preds.append(outputs.cpu())
                all_labels.append(labels.cpu())

        # Concatenate all batches
        full_preds = torch.cat(all_preds)
        full_labels = torch.cat(all_labels)

        # Calculate F0.5 Score
        val_f05 = utils.fbeta_score(full_preds, full_labels, beta=0.5)
        val_loss = running_loss / len(dataloader)

        return val_loss, val_f05

    def fit(self, epochs=config.NUM_EPOCHS, patience=7):
        set_seed(config.SEED)

        # Load Metadata
        train_df = pd.read_csv(os.path.join(config.METADATA_DIR, "train.csv"))
        val_df = pd.read_csv(os.path.join(config.METADATA_DIR, "validation.csv"))

        # Create Datasets
        train_dataset = dataset.InkDataset(
            train_df, mode="train", transforms=dataset.get_transforms("train")
        )
        val_dataset = dataset.InkDataset(
            val_df, mode="val", transforms=dataset.get_transforms("val")
        )

        # Create Dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

        print(f"Starting training on device: {self.device}")
        print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
        print(f"Validation Threshold: {self.validation_threshold}")

        epochs_no_improve = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(train_loader)

            # Validate
            val_loss, val_f05 = self.validate(val_loader)

            # Step Scheduler
            self.scheduler.step(val_f05)

            elapsed = time.time() - start_time

            # Print metrics with full precision
            current_lr = self.optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch}/{epochs} - Time: {elapsed:.2f}s - LR: {current_lr:.6f}"
            )
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val F0.5:   {val_f05}")

            # Validation Gating & Checkpointing
            # Only save if score improves AND exceeds threshold
            if val_f05 > self.best_score:
                if val_f05 > self.validation_threshold:
                    print(
                        f"  [+] New Best Score ({val_f05} > {self.best_score}) AND > Threshold. Saving model..."
                    )
                    self.best_score = val_f05
                    save_path = os.path.join(self.working_dir, "best_model.pth")
                    torch.save(self.model.state_dict(), save_path)
                    epochs_no_improve = 0
                else:
                    print(
                        f"  [!] Score improved ({val_f05}) but did not exceed threshold ({self.validation_threshold}). Model NOT saved."
                    )
                    # We update best_score to track progress, but don't save the file
                    self.best_score = val_f05
                    epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                print(f"  [.] No improvement. Patience: {epochs_no_improve}/{patience}")

            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break


def train_model():
    trainer = Trainer()
    trainer.fit()

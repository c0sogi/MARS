import os
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from library.config import Config
from library.dataset import INatDataset, get_transforms
from library.model import INatModel
from library.loss import FocalLoss


def set_seed(seed=Config.SEED):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(self):
        """
        Initializes the Trainer with model, optimizer, criterion, and scheduler.
        """
        set_seed()
        self.device = torch.device(Config.DEVICE)

        # Initialize Model
        print(f"Initializing model: {Config.MODEL_NAME}")
        self.model = INatModel(pretrained=Config.PRETRAINED)
        self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY,
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=Config.EPOCHS, eta_min=Config.MIN_LR
        )

        # Loss Function
        # Using Focal Loss to address class imbalance (Cite solution_lesson_node_00001, solution_lesson_node_00002)
        self.criterion = FocalLoss(gamma=Config.FOCAL_GAMMA)

        # Mixed Precision Scaler
        self.scaler = (
            torch.amp.GradScaler("cuda")
            if Config.USE_AMP and self.device.type == "cuda"
            else None
        )

    def get_dataloaders(self, debug=Config.DEBUG):
        """
        Creates and returns DataLoaders for train, validation, and test sets.
        """
        print("Loading metadata...")
        train_df = pd.read_csv(Config.TRAIN_METADATA)
        val_df = pd.read_csv(Config.VAL_METADATA)
        test_df = pd.read_csv(Config.TEST_METADATA)

        if debug:
            print(f"Debug mode: Sampling {Config.DEBUG_SAMPLE_SIZE} images.")
            train_df = train_df.head(Config.DEBUG_SAMPLE_SIZE)
            val_df = val_df.head(Config.DEBUG_SAMPLE_SIZE)
            test_df = test_df.head(Config.DEBUG_SAMPLE_SIZE)

        # Create Datasets
        train_dataset = INatDataset(train_df, transforms=get_transforms("train"))
        val_dataset = INatDataset(val_df, transforms=get_transforms("valid"))
        test_dataset = INatDataset(test_df, transforms=get_transforms("test"))

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=Config.BATCH_SIZE,
            shuffle=True,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=Config.VAL_BATCH_SIZE,
            shuffle=False,
            num_workers=Config.NUM_WORKERS,
            pin_memory=True,
        )

        return train_loader, val_loader, test_loader

    def train_epoch(self, train_loader):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for images, targets, _ in train_loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            # Mixed Precision Forward Pass
            if self.scaler:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            count += images.size(0)

        return running_loss / count

    def validate(self, val_loader):
        """
        Runs validation on the validation set.
        Returns average loss, Top-1 Accuracy, and Top-5 Accuracy.
        """
        self.model.eval()
        running_loss = 0.0
        correct_1 = 0
        correct_5 = 0
        count = 0

        with torch.no_grad():
            for images, targets, _ in val_loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                        loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, targets)

                running_loss += loss.item() * images.size(0)

                # Accuracy calculation
                _, pred = outputs.topk(5, 1, True, True)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))

                correct_1 += correct[:1].reshape(-1).float().sum().item()
                correct_5 += correct[:5].reshape(-1).float().sum().item()
                count += images.size(0)

        avg_loss = running_loss / count
        acc_1 = correct_1 / count * 100
        acc_5 = correct_5 / count * 100

        return avg_loss, acc_1, acc_5

    def fit(self, train_loader, val_loader, epochs=Config.EPOCHS):
        """
        Main training loop with Early Stopping.
        """
        print(f"Starting training for {epochs} epochs...")
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_acc1, val_acc5 = self.validate(val_loader)

            # Scheduler Step
            self.scheduler.step()

            elapsed = time.time() - start_time

            # Print Metrics (Full Precision)
            print(f"Epoch {epoch}/{epochs} - Time: {elapsed:.2f}s")
            print(f"  Train Loss: {train_loss}")
            print(f"  Val Loss:   {val_loss}")
            print(f"  Val Acc@1:  {val_acc1}%")
            print(f"  Val Acc@5:  {val_acc5}%")

            # Checkpoint & Early Stopping
            if val_loss < best_val_loss:
                print(
                    f"  Val Loss improved from {best_val_loss} to {val_loss}. Saving model..."
                )
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), Config.MODEL_CHECKPOINT)
                patience_counter = 0
            else:
                patience_counter += 1
                print(
                    f"  No improvement. Patience: {patience_counter}/{Config.EARLY_STOPPING_PATIENCE}"
                )

            if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                print("Early stopping triggered.")
                break

    def predict(self, test_loader):
        """
        Generates predictions for the test set and saves the submission file.
        """
        print("Loading best model for inference...")
        if not os.path.exists(Config.MODEL_CHECKPOINT):
            print("Warning: No checkpoint found. Using current model weights.")
        else:
            state_dict = torch.load(Config.MODEL_CHECKPOINT, map_location=self.device)
            self.model.load_state_dict(state_dict)

        self.model.eval()
        results = []

        print("Generating predictions...")
        with torch.no_grad():
            for images, _, image_ids in test_loader:
                images = images.to(self.device, non_blocking=True)

                if self.scaler:
                    with torch.amp.autocast("cuda"):
                        outputs = self.model(images)
                else:
                    outputs = self.model(images)

                # Get top 5 predictions
                # We need the indices of the top 5 scores
                _, top_indices = torch.topk(outputs, k=5, dim=1)

                top_indices = top_indices.cpu().numpy()
                image_ids = image_ids.numpy()

                for img_id, preds in zip(image_ids, top_indices):
                    # Format: "cat_id1 cat_id2 cat_id3 cat_id4 cat_id5"
                    pred_str = " ".join(map(str, preds))
                    results.append({"id": img_id, "predicted": pred_str})

        # Save Submission
        submission_df = pd.DataFrame(results)
        submission_df.to_csv(Config.SUBMISSION_FILE, index=False)
        print(f"Submission saved to {Config.SUBMISSION_FILE}")
        print(submission_df.head())

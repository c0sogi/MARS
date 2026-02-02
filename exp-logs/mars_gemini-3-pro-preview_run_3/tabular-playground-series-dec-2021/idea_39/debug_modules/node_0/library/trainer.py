import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from library.utils import seed_everything, get_device
from library.data_loader import get_dataloaders
from library.model import (
    WideAsymmetricDCNResNet,
    train_one_epoch,
    validate,
    predict_test,
)


class Trainer:
    """
    Manages the training, evaluation, and submission generation for the Forest Cover Type task.
    Implements a Stabilized Wide Asymmetric Parallel Vector-DCN-ResNet strategy.
    """

    def __init__(
        self,
        epochs=60,
        batch_size=4096,
        learning_rate=1e-3,
        warmup_epochs=5,
        patience=15,
        weight_decay=1e-2,
        dropout=0.3,
        hidden_dim=1024,
        cache_dir="./working/idea_39/",
        metadata_dir="./metadata",
        output_dir="./submission",
        seed=42,
    ):
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.cache_dir = cache_dir
        self.metadata_dir = metadata_dir
        self.output_dir = output_dir
        self.seed = seed

        self.device = None
        self.model = None
        self.best_model_state = None
        self.input_dim = None
        self.num_classes = None

    def train(self):
        """
        Executes the training pipeline including data loading, model initialization,
        optimization loop with warmup and scheduling, and early stopping.
        """
        # 1. Setup
        seed_everything(self.seed)
        self.device = get_device()
        print(f"Using device: {self.device}")

        # 2. Load Data
        print("Loading data...")
        # get_dataloaders handles the caching logic internally via library.utils.get_data
        train_loader, val_loader, test_loader, self.input_dim, self.num_classes = (
            get_dataloaders(
                batch_size=self.batch_size,
                load_cached_data=True,
                cache_dir=self.cache_dir,
                metadata_dir=self.metadata_dir,
            )
        )
        print(f"Input Dim: {self.input_dim}, Num Classes: {self.num_classes}")

        # 3. Initialize Model
        self.model = WideAsymmetricDCNResNet(
            input_dim=self.input_dim,
            num_classes=self.num_classes,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        # 4. Optimizer & Loss
        # AdamW with Decoupled Weight Decay
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        # Scheduler: ReduceLROnPlateau
        # Note: We apply this after the warmup phase
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.1, patience=5, verbose=True
        )

        # 5. Training Loop
        best_acc = 0.0
        self.best_model_state = copy.deepcopy(self.model.state_dict())
        early_stop_counter = 0

        print("Starting training...")
        for epoch in range(self.epochs):
            # --- Linear Warmup Logic ---
            if epoch < self.warmup_epochs:
                # Ramp from 1e-6 to base learning_rate
                start_lr = 1e-6
                progress = (epoch + 1) / self.warmup_epochs
                current_lr = start_lr + (self.learning_rate - start_lr) * progress

                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

            # --- Train & Validate ---
            train_loss, train_acc = train_one_epoch(
                self.model, train_loader, optimizer, criterion, self.device
            )
            val_loss, val_acc = validate(self.model, val_loader, criterion, self.device)

            # --- Scheduler Step ---
            # Apply scheduler only after warmup is complete
            if epoch >= self.warmup_epochs:
                scheduler.step(val_acc)

            # --- Logging ---
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch+1}/{self.epochs} | LR: {current_lr:.10f} | "
                f"Train Loss: {train_loss:.10f} Acc: {train_acc:.10f} | "
                f"Val Loss: {val_loss:.10f} Acc: {val_acc:.10f}"
            )

            # --- Checkpointing & Early Stopping ---
            if val_acc > best_acc:
                best_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Best Validation Accuracy: {best_acc:.10f}")

        # Restore best model for prediction
        self.model.load_state_dict(self.best_model_state)

        # Generate submission immediately after training
        self.generate_submission(test_loader)

    def generate_submission(self, test_loader=None):
        """
        Generates predictions on the test set and saves them to a CSV file.
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained or initialized.")

        if test_loader is None:
            # Reload loaders if necessary (though usually passed or stored)
            _, _, test_loader, _, _ = get_dataloaders(
                batch_size=self.batch_size,
                load_cached_data=True,
                cache_dir=self.cache_dir,
                metadata_dir=self.metadata_dir,
            )

        print("Generating predictions...")
        ids, preds = predict_test(self.model, test_loader, self.device)

        # Map 0-indexed predictions back to 1-indexed labels
        # Dataset utils mapped 1-7 -> 0-6. We reverse this: 0-6 -> 1-7.
        preds_mapped = [p + 1 for p in preds]

        os.makedirs(self.output_dir, exist_ok=True)
        submission_path = os.path.join(self.output_dir, "submission.csv")

        df_sub = pd.DataFrame({"Id": ids, "Cover_Type": preds_mapped})
        df_sub.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from library.config import Config
from library.model import DeepParallelDCNResNet
from library.data_utils import get_datasets


def set_seed(seed=42):
    """Sets the random seed for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Manages the training, evaluation, and prediction processes.
    """

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer: AdamW (Decoupled Weight Decay)
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY,
        )

        # Scheduler: ReduceLROnPlateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="max",
            factor=self.config.SCHEDULER_FACTOR,
            patience=self.config.SCHEDULER_PATIENCE,
        )

    def train_one_epoch(self, train_loader):
        """Runs one epoch of training."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def evaluate(self, val_loader):
        """Evaluates the model on the validation set."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return epoch_loss, epoch_acc

    def fit(self, train_loader, val_loader):
        """
        Runs the full training loop with Early Stopping and Checkpointing.
        """
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0

        print("Starting training...")
        for epoch in range(self.config.EPOCHS):
            train_loss, train_acc = self.train_one_epoch(train_loader)
            val_loss, val_acc = self.evaluate(val_loader)

            # Print full precision metrics
            print(
                f"Epoch {epoch+1}/{self.config.EPOCHS} | "
                f"Train Loss: {train_loss} | Train Acc: {train_acc} | "
                f"Val Loss: {val_loss} | Val Acc: {val_acc}"
            )

            # Step the scheduler based on validation accuracy
            self.scheduler.step(val_acc)

            # Early Stopping and Checkpointing logic
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = copy.deepcopy(self.model.state_dict())
                # Save the best model immediately
                torch.save(best_model_state, self.config.MODEL_SAVE_PATH)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        print(f"Best Validation Accuracy: {best_val_acc}")

        # Restore best model weights for inference
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, test_loader):
        """Generates predictions for the test set."""
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                # TensorDataset for test returns a tuple (inputs,)
                inputs = inputs[0].to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
        return np.array(all_preds)


def run_training(load_cached_data=True, debug=False):
    """
    Main execution function to load data, train the model, and generate submission.
    """
    # Ensure reproducibility
    set_seed(Config.SEED)

    # Ensure directories exist
    Config.create_directories()

    # 1. Load Data using the utility function
    # This handles caching and preprocessing internally
    train_dataset, val_dataset, test_dataset, test_ids, classes = get_datasets(
        load_cached_data=load_cached_data, debug=debug
    )

    # 2. Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,
    )

    # 3. Initialize Model
    # Determine input dimension from the dataset features
    input_dim = train_dataset[0][0].shape[0]
    num_classes = len(classes)
    device = torch.device(Config.DEVICE)

    model = DeepParallelDCNResNet(
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=Config.HIDDEN_DIM,
        num_res_blocks=Config.NUM_RES_BLOCKS,
        dropout_rate=Config.DROPOUT_RATE,
    ).to(device)

    # 4. Initialize Trainer and Start Training
    trainer = Trainer(model, device, Config)
    trainer.fit(train_loader, val_loader)

    # 5. Generate Predictions
    print("Generating predictions on test set...")
    raw_preds = trainer.predict(test_loader)
    final_preds = classes[raw_preds]

    # 6. Save Submission
    submission = pd.DataFrame({Config.ID_COL: test_ids, Config.TARGET_COL: final_preds})
    submission.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

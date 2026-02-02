import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import PatentBert, train_one_epoch, evaluate, generate_submission


class Trainer:
    """
    Trainer module to encapsulate training, evaluation, and prediction loops.
    """

    def __init__(self):
        self.device = torch.device(Config.DEVICE)
        set_seed(Config.SEED)

    def train_epoch(self, model, dataloader, criterion, optimizer):
        """
        Performs one epoch of training.
        """
        return train_one_epoch(model, dataloader, criterion, optimizer, self.device)

    def evaluate(self, model, dataloader, criterion):
        """
        Evaluates the model on the validation set.
        """
        return evaluate(model, dataloader, criterion, self.device)

    def predict(self, model, test_loader):
        """
        Generates predictions for the test set.
        """
        generate_submission(model, test_loader, self.device, Config.SUBMISSION_PATH)

    def train(self, epochs=Config.EPOCHS, debug=Config.DEBUG, load_cached_data=True):
        """
        Orchestrates the training process, including data loading, model initialization,
        training loop, early stopping, and final prediction.
        """
        # Update Config based on arguments for flexibility
        Config.EPOCHS = epochs
        Config.DEBUG = debug

        # Load Data
        # Caching logic is handled internally by get_dataloaders via load_cached_data flag
        train_loader, val_loader, test_loader = get_dataloaders(
            load_cached_data=load_cached_data
        )

        # Initialize Model
        model = PatentBert(Config.MODEL_NAME).to(self.device)

        # Initialize Optimizer and Loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # Training Loop variables
        best_val_loss = float("inf")
        patience_counter = 0

        print(f"Starting training on {self.device}...")

        for epoch in range(Config.EPOCHS):
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)

            # Validate
            val_loss, val_pearson = self.evaluate(model, val_loader, criterion)

            # Print metrics with full precision
            print(
                f"Epoch {epoch+1}/{Config.EPOCHS} | Train Loss: {train_loss} | Val Loss: {val_loss} | Val Pearson: {val_pearson}"
            )

            # Early Stopping Check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        # Load best model for inference
        if os.path.exists(Config.MODEL_SAVE_PATH):
            print(f"Loading best model from {Config.MODEL_SAVE_PATH}...")
            model.load_state_dict(
                torch.load(Config.MODEL_SAVE_PATH, map_location=self.device)
            )
        else:
            print("Warning: No model file saved. Using current model state.")

        # Generate Submission
        print("Generating submission...")
        self.predict(model, test_loader)

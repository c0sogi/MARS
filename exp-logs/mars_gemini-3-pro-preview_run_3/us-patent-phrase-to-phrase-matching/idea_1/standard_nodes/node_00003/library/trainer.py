import os
import torch
import torch.nn as nn
import torch.optim as optim
from library.config import Config, set_seed
from library.data_loader import get_dataloaders
from library.model import SiameseDAN, train_one_epoch, evaluate, generate_submission


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
        (
            train_loader,
            val_loader,
            test_loader,
            vocab_size,
            num_contexts,
            embedding_matrix,
        ) = get_dataloaders(load_cached_data=load_cached_data)

        # Initialize Model
        model = SiameseDAN(
            vocab_size=vocab_size,
            num_contexts=num_contexts,
            embedding_dim=Config.EMBEDDING_DIM,
            hidden_dim=Config.HIDDEN_DIM,
            context_dim=Config.CONTEXT_EMBEDDING_DIM,
            dropout_rate=Config.DROPOUT,
            pretrained_embeddings=embedding_matrix,
        ).to(self.device)

        # Initialize Optimizer and Loss
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

        # Training Loop variables
        best_val_loss = float("inf")
        best_val_pearson = 0.0
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
                best_val_pearson = val_pearson
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), Config.MODEL_SAVE_PATH)
            else:
                patience_counter += 1
                if patience_counter >= Config.PATIENCE:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break

        return best_val_pearson

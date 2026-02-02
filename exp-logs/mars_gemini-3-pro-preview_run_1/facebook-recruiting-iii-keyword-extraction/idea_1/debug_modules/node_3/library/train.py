import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from library.config import Config
from library.utils import seed_everything, calculate_f1, save_checkpoint
from library.model import BiGRUClassifier
from library.data_loader import get_dataloaders


class Trainer:
    """
    Manages the training and validation lifecycle of the model.
    """

    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_one_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        running_loss = 0.0
        count = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(inputs)

            # Compute loss
            loss = self.criterion(logits, targets)

            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            count += inputs.size(0)

        avg_loss = running_loss / count
        return avg_loss

    def validate(self):
        """
        Runs validation on the validation set.
        Returns average loss and Mean F1 Score.
        """
        self.model.eval()
        running_loss = 0.0
        count = 0

        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(inputs)
                loss = self.criterion(logits, targets)

                running_loss += loss.item() * inputs.size(0)
                count += inputs.size(0)

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Convert to binary predictions based on threshold
                preds = (probs > Config.PREDICTION_THRESHOLD).float()

                # Move to CPU for metric calculation
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        avg_loss = running_loss / count

        # Concatenate all batches
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # Calculate F1 Score
        val_f1 = calculate_f1(all_targets, all_preds, average="samples")

        return avg_loss, val_f1

    def fit(
        self, num_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE
    ):
        """
        Main training loop with Early Stopping.
        """
        best_f1 = -1.0
        patience_counter = 0

        print(f"Starting training on device: {self.device}")

        for epoch in range(1, num_epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_one_epoch(epoch)

            # Validate
            val_loss, val_f1 = self.validate()

            elapsed = time.time() - start_time

            print(f"Epoch {epoch}/{num_epochs} | Time: {elapsed:.2f}s")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val F1: {val_f1}")  # Printing full precision as requested

            # Checkpoint and Early Stopping
            if val_f1 > best_f1:
                print(
                    f"Validation F1 improved from {best_f1} to {val_f1}. Saving model..."
                )
                best_f1 = val_f1
                save_checkpoint(self.model, self.optimizer, epoch, val_f1)
                patience_counter = 0
            else:
                patience_counter += 1
                print(f"No improvement. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        print(f"Training complete. Best Val F1: {best_f1}")
        return best_f1


def run_training(debug=Config.DEBUG):
    """
    Sets up the environment, loads data, and runs the training process.
    """
    # 1. Set Seed
    seed_everything(Config.SEED)

    # 2. Load Data
    train_loader, val_loader, tokenizer, tag_encoder = get_dataloaders(debug=debug)

    # 3. Initialize Model
    # We use the actual vocabulary size from the fitted tokenizer
    actual_vocab_size = len(tokenizer.vocab)
    print(f"Initializing model with vocab size: {actual_vocab_size}")

    model = BiGRUClassifier(
        vocab_size=actual_vocab_size,
        embed_dim=Config.EMBED_DIM,
        hidden_dim=Config.HIDDEN_DIM,
        num_layers=Config.NUM_LAYERS,
        num_classes=Config.NUM_TAGS,
        dropout=Config.DROPOUT,
    )

    device = torch.device(Config.DEVICE)
    model.to(device)

    # 4. Setup Optimizer and Loss
    # BCEWithLogitsLoss combines Sigmoid and BCELoss, numerically stable
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
    )

    # 6. Start Training
    trainer.fit(num_epochs=Config.NUM_EPOCHS, patience=Config.EARLY_STOPPING_PATIENCE)

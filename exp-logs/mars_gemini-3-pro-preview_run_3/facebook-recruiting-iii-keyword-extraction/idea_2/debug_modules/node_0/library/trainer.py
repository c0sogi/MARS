import time
import torch
import numpy as np
import torch.nn as nn
from library.config import Config
from library.utils import calculate_metrics, save_checkpoint, create_submission


class Trainer:
    """
    Trainer class to manage training, validation, and inference for the NBOW model.
    """

    def __init__(self, model, optimizer, criterion, device=Config.DEVICE):
        """
        Args:
            model (nn.Module): The PyTorch model.
            optimizer (torch.optim.Optimizer): The optimizer.
            criterion (nn.Module): The loss function (e.g., BCEWithLogitsLoss).
            device (torch.device): Device to run on (CPU or GPU).
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        """
        Runs one epoch of training.

        Args:
            dataloader (DataLoader): Training data loader.

        Returns:
            float: Average training loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(dataloader)

        for batch in dataloader:
            # Unpack batch: (flat_tokens, offsets, labels)
            # Note: collate_fn returns flat_inputs, offsets, targets
            input_ids, offsets, targets = batch

            input_ids = input_ids.to(self.device)
            offsets = offsets.to(self.device)
            targets = targets.to(self.device)

            self.optimizer.zero_grad()

            # Forward pass
            logits = self.model(input_ids, offsets)
            loss = self.criterion(logits, targets)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / num_batches

    def evaluate(self, dataloader):
        """
        Evaluates the model on the validation set.

        Args:
            dataloader (DataLoader): Validation data loader.

        Returns:
            tuple: (average_loss, f1_score)
        """
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in dataloader:
                input_ids, offsets, targets = batch

                input_ids = input_ids.to(self.device)
                offsets = offsets.to(self.device)
                targets = targets.to(self.device)

                logits = self.model(input_ids, offsets)
                loss = self.criterion(logits, targets)

                total_loss += loss.item()

                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(logits)

                # Move to CPU for metric calculation
                all_preds.append(probs.cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(dataloader)

        # Concatenate all batches
        y_pred = torch.cat(all_preds)
        y_true = torch.cat(all_targets)

        # Calculate F1 Score
        # calculate_metrics applies the threshold (default 0.5) internally
        f1 = calculate_metrics(y_true, y_pred)

        return avg_loss, f1

    def fit(
        self,
        train_loader,
        val_loader,
        epochs=Config.NUM_EPOCHS,
        patience=Config.EARLY_STOPPING_PATIENCE,
    ):
        """
        Runs the full training loop with early stopping.

        Args:
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            epochs (int): Maximum number of epochs.
            patience (int): Early stopping patience.

        Returns:
            float: Best validation F1 score achieved.
        """
        best_val_f1 = 0.0
        patience_counter = 0

        print(f"Starting training on {self.device} for {epochs} epochs.")

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # Train
            train_loss = self.train_epoch(train_loader)

            # Validate
            val_loss, val_f1 = self.evaluate(val_loader)

            end_time = time.time()
            duration = (end_time - start_time) / 60.0  # Minutes

            # Print metrics (Full precision as requested)
            print(f"Epoch {epoch}/{epochs} | Time: {duration} mins")
            print(f"Train Loss: {train_loss}")
            print(f"Val Loss: {val_loss}")
            print(f"Val F1: {val_f1}")

            # Checkpoint & Early Stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                save_checkpoint(self.model, self.optimizer, epoch, val_f1)
                print("New best model saved.")
            else:
                patience_counter += 1
                print(f"EarlyStopping counter: {patience_counter} out of {patience}")

            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

        return best_val_f1

    def predict(self, test_loader):
        """
        Generates predictions for the test set.

        Args:
            test_loader (DataLoader): Test data loader.

        Returns:
            tuple: (ids, probs) where ids is a numpy array of question IDs
                   and probs is a numpy array of predicted probabilities.
        """
        self.model.eval()
        all_probs = []
        all_ids = []

        with torch.no_grad():
            for batch in test_loader:
                # Unpack batch: (flat_tokens, offsets, ids)
                input_ids, offsets, ids = batch

                input_ids = input_ids.to(self.device)
                offsets = offsets.to(self.device)

                logits = self.model(input_ids, offsets)
                probs = torch.sigmoid(logits)

                all_probs.append(probs.cpu().numpy())
                all_ids.append(ids.numpy())

        return np.concatenate(all_ids), np.concatenate(all_probs)


def run_inference(
    trainer, test_loader, mlb, threshold=0.5, output_path=Config.SUBMISSION_PATH
):
    """
    Helper function to run inference and save submission file.

    Args:
        trainer (Trainer): Trained trainer instance.
        test_loader (DataLoader): Test data loader.
        mlb (MultiLabelBinarizer): Fitted tag encoder.
        threshold (float): Probability threshold for binary classification.
        output_path (str): Path to save the submission CSV.
    """
    print("Generating predictions on test set...")
    ids, probs = trainer.predict(test_loader)

    print(f"Saving submission to {output_path}...")
    create_submission(ids, probs, mlb, threshold=threshold, output_path=output_path)
    print("Submission saved successfully.")

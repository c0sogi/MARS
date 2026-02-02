import time
import torch
import torch.nn as nn
import torch.optim as optim
import math
import os
from library.config import Config


class Trainer:
    """
    Trainer class for the Seq2Seq Text Normalization model.
    Handles training loops, validation, early stopping, and checkpointing.
    """

    def __init__(self, model, train_loader, val_loader, device):
        """
        Args:
            model (nn.Module): The Seq2Seq model.
            train_loader (DataLoader): Training data loader.
            val_loader (DataLoader): Validation data loader.
            device (torch.device): Device to run training on.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Initialize Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=Config.LEARNING_RATE)

        # Initialize Loss Function
        # We ignore the padding index so it doesn't contribute to the loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=Config.PAD_IDX)

    def train_epoch(self, epoch_idx):
        """
        Runs one epoch of training.
        """
        self.model.train()
        epoch_loss = 0

        for batch_idx, batch in enumerate(self.train_loader):
            src = batch["src"].to(self.device)
            tgt = batch["tgt"].to(self.device)
            src_len = batch["src_len"].to(
                "cpu"
            )  # PackedSequence/LSTM often prefers lengths on CPU or specific format

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            # output: [batch_size, tgt_len, vocab_size]
            output = self.model(
                src, src_len, tgt, teacher_forcing_ratio=Config.TEACHER_FORCING_RATIO
            )

            # Calculate Loss
            # output shape: [batch_size, tgt_len, vocab_size]
            # tgt shape:    [batch_size, tgt_len]

            # We slice off the 0-th index of output and target.
            # Output at index 0 is all zeros (initialized in model wrapper).
            # Target at index 0 is <sos>.
            # We want output[:, 1, :] to predict tgt[:, 1]

            output_dim = output.shape[-1]

            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)

            loss = self.criterion(output, tgt)

            # Backward pass
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), Config.CLIP_GRAD)

            # Update weights
            self.optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(self.train_loader)

    def evaluate(self):
        """
        Runs evaluation on the validation set.
        """
        self.model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch["src"].to(self.device)
                tgt = batch["tgt"].to(self.device)
                src_len = batch["src_len"].to("cpu")

                # Turn off teacher forcing for evaluation (ratio = 0)
                output = self.model(src, src_len, tgt, teacher_forcing_ratio=0.0)

                output_dim = output.shape[-1]

                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)

                loss = self.criterion(output, tgt)

                epoch_loss += loss.item()

        return epoch_loss / len(self.val_loader)

    def fit(self):
        """
        Main training loop with Early Stopping.
        """
        best_valid_loss = float("inf")
        patience_counter = 0

        print(f"Starting training for {Config.N_EPOCHS} epochs...")
        print(f"Device: {self.device}")

        for epoch in range(Config.N_EPOCHS):
            start_time = time.time()

            train_loss = self.train_epoch(epoch)
            valid_loss = self.evaluate()

            end_time = time.time()
            epoch_mins, epoch_secs = divmod(end_time - start_time, 60)

            print(f"Epoch: {epoch+1:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s")
            print(f"\tTrain Loss: {train_loss}")
            print(f"\t Val. Loss: {valid_loss}")

            # Checkpoint and Early Stopping
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), Config.MODEL_SAVE_PATH)
                print(
                    f"\tValidation loss decreased. Model saved to {Config.MODEL_SAVE_PATH}"
                )
            else:
                patience_counter += 1
                print(
                    f"\tValidation loss did not decrease. Patience: {patience_counter}/{Config.PATIENCE}"
                )

                if patience_counter >= Config.PATIENCE:
                    print("Early stopping triggered.")
                    break
